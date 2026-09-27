"""End-to-end CLI on synthetic tifxyz meshes written to disk."""

import json

import numpy as np
import pytest

from handcheck import cli
from handcheck import synth as S
from handcheck.tifxyz import TifxyzMesh, write_tifxyz

CENTRE = (70.0, 70.0)


def _spiral():
    return S.Spiral(centre=CENTRE, r_outer=62.0, pitch=9.0, turns=4.5, phi_start=np.radians(-20.0), w=1)


@pytest.fixture()
def folder(tmp_path):
    g = S.spiral_sheet_grid(_spiral(), 1, (40, 40), 1.5, 0.0, s0=-30.0, z0=20.0, phi_ref=np.radians(35.0))
    write_tifxyz(tmp_path / "meshes" / "as_stored.tifxyz", TifxyzMesh(points=g, scale=(1 / 1.5, 1 / 1.5), uuid="a"))
    write_tifxyz(tmp_path / "meshes" / "u_mirrored.tifxyz",
                 TifxyzMesh(points=np.ascontiguousarray(g[:, ::-1]), scale=(1 / 1.5, 1 / 1.5), uuid="b"))
    (tmp_path / "props.json").write_text(json.dumps(dict(left_handed_coordinates=False,
                                                         z_direction_is_top_to_bottom=True)))
    (tmp_path / "umb.json").write_text(json.dumps(
        {"control_points": [{"x": CENTRE[0], "y": CENTRE[1], "z": 0.0}, {"x": CENTRE[0], "y": CENTRE[1], "z": 110.0}]}))
    return tmp_path


def test_cli_predicts_mirror_state(folder, capsys):
    out = folder / "out.json"
    rc = cli.main([str(folder / "meshes"), "--props", str(folder / "props.json"), "--axes", "zyx",
                   "--umbilicus", str(folder / "umb.json"), "--json", str(out), "--csv", str(folder / "out.csv")])
    assert rc == 0
    rows = {r["mesh"]: r for r in json.loads(out.read_text())["meshes"]}
    assert rows["as_stored.tifxyz"]["state"] == "OK" and rows["as_stored.tifxyz"]["mirrored_as_stored"] is False
    assert rows["u_mirrored.tifxyz"]["state"] == "OK" and rows["u_mirrored.tifxyz"]["mirrored_as_stored"] is True
    assert rows["as_stored.tifxyz"]["confidence"] == "HIGH"
    assert "frame not checked" in rows["as_stored.tifxyz"]["flags"]
    header = (folder / "out.csv").read_text().splitlines()[0].split(",")
    assert header == cli.CSV_FIELDS


def test_cli_parity_unknown_and_strict(folder):
    rc = cli.main([str(folder / "meshes"), "--props", str(folder / "props.json"), "--strict"])  # no axes given
    assert rc == 3


def test_cli_frame_gate(folder):
    (folder / "frame.json").write_text(json.dumps({"as_stored.tifxyz": {"frac_zero_ct": 0.4, "frac_outside": 0.0}}))
    out = folder / "o.json"
    cli.main([str(folder / "meshes"), "--props", str(folder / "props.json"), "--axes", "zyx",
              "--umbilicus", str(folder / "umb.json"), "--frame", str(folder / "frame.json"), "--json", str(out)])
    rows = {r["mesh"]: r for r in json.loads(out.read_text())["meshes"]}
    assert rows["as_stored.tifxyz"]["state"] == "FRAME_UNVERIFIED"
    assert rows["as_stored.tifxyz"]["mirrored_as_stored"] is None


def test_cli_ct_zarr_frame_check(folder):
    zarr = pytest.importorskip("zarr")
    vol = np.zeros((8, 10, 10), np.uint8)  # level 4 of a 128 x 160 x 160 volume; ones where the sheet is
    vol[1:5, 1:9, 1:9] = 1
    zarr.create_array(store=str(folder / "vol.zarr" / "4"), data=vol)
    out = folder / "z.json"
    rc = cli.main([str(folder / "meshes"), "--props", str(folder / "props.json"), "--axes", "zyx",
                   "--umbilicus", str(folder / "umb.json"), "--ct-zarr", str(folder / "vol.zarr"), "--json", str(out)])
    assert rc == 0
    for r in json.loads(out.read_text())["meshes"]:
        assert r["frame"]["frac_outside"] == 0.0 and r["frame"]["frac_zero_ct"] < 0.05, r["frame"]
        assert r["state"] == "OK"


def test_load_centre_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([[1, 2, 0], [1, 2, 10]]))
    (tmp_path / "b.json").write_text(json.dumps({"z": [0, 10], "x": [1, 1], "y": [2, 2]}))
    for n in ("a.json", "b.json"):
        c = cli.load_centre(str(tmp_path / n))
        assert c(np.array([5.0]))[0][0] == 1.0 and c(np.array([5.0]))[1][0] == 2.0
