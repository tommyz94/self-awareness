from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _write_real_humanoid_asset_pack(target_xml: Path) -> None:
    source_xml = (
        Path(__file__).resolve().parent.parent / "assets" / "mujoco" / "humanoid_real.xml"
    )
    target_xml.write_text(source_xml.read_text())

    root = ET.parse(target_xml).getroot()
    compiler = root.find("./compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for asset_node in root.findall("./asset/*[@file]"):
        relative_path = Path(asset_node.get("file"))
        if asset_node.tag == "mesh" and meshdir:
            asset_path = target_xml.parent / meshdir / relative_path
        elif asset_node.tag == "texture" and texturedir:
            asset_path = target_xml.parent / texturedir / relative_path
        else:
            asset_path = target_xml.parent / relative_path
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(b"placeholder")


def test_mujoco_builder_supports_pelvis_root_assets_and_absolutizes_files(
    tmp_path,
    monkeypatch,
):
    import env.scenes.mujoco_builder as builder

    temp_xml = tmp_path / "humanoid_real.xml"
    _write_real_humanoid_asset_pack(temp_xml)

    monkeypatch.setattr(builder, "_HUMANOID_XML_PATH", temp_xml)

    fragments = builder.build_humanoid_scene(
        mirror_present=True,
        spawn_pos=(-10.0, -1.0, 2.5),
        camera_height=0.5,
    )

    assert 'name="robot_pelvis"' in fragments.robot_body_xml
    assert 'name="mirror_pelvis"' in fragments.mirror_body_xml
    assert 'name="first_person"' in fragments.robot_body_xml
    assert str((tmp_path / "textures" / "skin_albedo.png").resolve().as_posix()) in fragments.asset_xml


def test_select_humanoid_xml_path_prefers_real_asset_when_dependencies_exist(
    tmp_path,
    monkeypatch,
):
    import env.scenes.mujoco_builder as builder

    legacy_xml = tmp_path / "humanoid.xml"
    legacy_xml.write_text("<mujoco model='legacy'><worldbody><body name='torso'/></worldbody></mujoco>")
    real_xml = tmp_path / "humanoid_real.xml"
    _write_real_humanoid_asset_pack(real_xml)

    monkeypatch.setattr(builder, "_LEGACY_HUMANOID_XML_PATH", legacy_xml)
    monkeypatch.setattr(builder, "_PREFERRED_HUMANOID_XML_PATH", real_xml, raising=False)

    assert builder._select_humanoid_xml_path() == real_xml
