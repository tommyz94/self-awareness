from __future__ import annotations

from experiments.specs.builtin_helpers import build_cube_run_setup, build_mirror_room_scene


def test_two_room_mujoco_xml_uses_textured_materials_and_explicit_lighting():
    from env.backends.mujoco import (
        _FLOOR_TEXTURE_PATH,
        _WALL_TEXTURE_PATH,
        _build_two_room_xml,
    )

    run_setup = build_cube_run_setup(build_mirror_room_scene(mirror_enabled=True))
    xml, _, _ = _build_two_room_xml(run_setup.scene_spec)

    assert _FLOOR_TEXTURE_PATH.exists()
    assert _WALL_TEXTURE_PATH.exists()

    assert 'texture name="floor_wood_tex"' in xml
    assert 'texture name="wall_plaster_tex"' in xml
    assert 'material name="floor_wood_mat"' in xml
    assert 'material name="wall_plaster_mat"' in xml

    assert 'material="floor_wood_mat"' in xml
    assert 'material="wall_plaster_mat"' in xml

    assert "<quality" in xml
    assert 'shadowsize="' in xml
    assert 'offsamples="' in xml
    assert 'offwidth="1600"' in xml
    assert 'offheight="1200"' in xml

    assert 'name="key_light"' in xml
    assert 'name="fill_light"' in xml
    assert 'castshadow="false"' in xml
