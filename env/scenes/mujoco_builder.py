from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "mujoco"
_LEGACY_HUMANOID_XML_PATH = _ASSETS_DIR / "humanoid.xml"
_PREFERRED_HUMANOID_XML_PATH = _ASSETS_DIR / "humanoid_real.xml"
_HUMANOID_ROT_EULER = "-90 0 0"
_FIRST_PERSON_CAMERA = {
    "name": "first_person",
    "pos": "0.15 1.45 0",
    "zaxis": "-1 0 0",
    "fovy": "110",
}
_THIRD_PERSON_CAMERA = {
    "name": "third_person",
    "pos": "-3 2 0",
    "xyaxes": "0 0 1 0.306 0.914 0",
}
_AGENT_FLAT_MATERIAL_NAME = "agent_flat_mat"
_AGENT_FLAT_MATERIAL_XML = (
    f'<material name="{_AGENT_FLAT_MATERIAL_NAME}" '
    'rgba="1 1 1 1" reflectance="0" shininess="0" specular="0" emission="0.0"/>'
)


@dataclass(frozen=True)
class HumanoidSceneFragments:
    asset_xml: str
    default_xml: str
    robot_body_xml: str
    mirror_body_xml: str
    robot_geom_names: list[str]
    mirror_geom_names: list[str]


def _asset_file_paths(humanoid_xml_path: Path) -> list[Path]:
    if not humanoid_xml_path.exists():
        return []

    root = ET.parse(humanoid_xml_path).getroot()
    asset_root = root.find("./asset")
    if asset_root is None:
        return []

    return [
        _resolve_asset_file_path(humanoid_xml_path, root, asset_node)
        for asset_node in asset_root.findall(".//*[@file]")
    ]


def _select_humanoid_xml_path() -> Path:
    if _PREFERRED_HUMANOID_XML_PATH.exists():
        asset_paths = _asset_file_paths(_PREFERRED_HUMANOID_XML_PATH)
        if all(path.exists() for path in asset_paths):
            return _PREFERRED_HUMANOID_XML_PATH
    return _LEGACY_HUMANOID_XML_PATH


def _resolve_asset_file_path(humanoid_xml_path: Path, root: ET.Element, asset_node: ET.Element) -> Path:
    raw_path = Path(asset_node.get("file", ""))
    if raw_path.is_absolute():
        return raw_path

    compiler = root.find("./compiler")
    asset_dir = humanoid_xml_path.parent
    if compiler is not None:
        if asset_node.tag == "mesh" and compiler.get("meshdir"):
            return (asset_dir / compiler.get("meshdir") / raw_path).resolve()
        if asset_node.tag == "texture" and compiler.get("texturedir"):
            return (asset_dir / compiler.get("texturedir") / raw_path).resolve()
    return (asset_dir / raw_path).resolve()


def _absolutize_asset_file_paths(humanoid_xml_path: Path, root: ET.Element) -> None:
    asset_root = root.find("./asset")
    if asset_root is None:
        return

    for asset_node in asset_root.findall(".//*[@file]"):
        resolved_path = _resolve_asset_file_path(humanoid_xml_path, root, asset_node)
        asset_node.set("file", resolved_path.as_posix())


def _find_humanoid_root_body(humanoid_root: ET.Element) -> ET.Element:
    worldbody = humanoid_root.find("./worldbody")
    if worldbody is None:
        raise ValueError("Humanoid asset is missing a <worldbody> section.")

    for body_name in ("torso", "pelvis"):
        body = worldbody.find(f"./body[@name='{body_name}']")
        if body is not None:
            return body

    for body in worldbody.findall("./body"):
        if body.find("./freejoint") is not None:
            return body

    first_body = worldbody.find("./body")
    if first_body is not None:
        return first_body

    raise ValueError("Humanoid asset is missing a root body under worldbody.")


_HUMANOID_XML_PATH = _select_humanoid_xml_path()


def _load_humanoid_root() -> ET.Element:
    if not _HUMANOID_XML_PATH.exists():
        raise FileNotFoundError(f"MuJoCo humanoid asset not found at {_HUMANOID_XML_PATH}")
    root = ET.parse(_HUMANOID_XML_PATH).getroot()
    _absolutize_asset_file_paths(_HUMANOID_XML_PATH, root)
    return root


def _serialize_optional(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return ET.tostring(element, encoding="unicode")


def _inject_agent_flat_material(asset_xml: str) -> str:
    if not asset_xml:
        return f"<asset>{_AGENT_FLAT_MATERIAL_XML}</asset>"
    if _AGENT_FLAT_MATERIAL_NAME in asset_xml:
        return asset_xml
    closing_tag = "</asset>"
    if closing_tag not in asset_xml:
        return asset_xml
    return asset_xml.replace(closing_tag, f"{_AGENT_FLAT_MATERIAL_XML}{closing_tag}")


def _remove_tag_recursive(element: ET.Element, tag_name: str) -> None:
    for child in list(element):
        if child.tag == tag_name:
            element.remove(child)
            continue
        _remove_tag_recursive(child, tag_name)


def _prefix_named_elements(element: ET.Element, prefix: str) -> list[str]:
    geom_names: list[str] = []
    for node in element.iter():
        name = node.get("name")
        if name:
            prefixed_name = f"{prefix}{name}"
            node.set("name", prefixed_name)
            if node.tag == "geom":
                geom_names.append(prefixed_name)
        if node.tag == "geom":
            # Force a neutral material so runtime geom_rgba maps to a stable perceived color.
            node.set("material", _AGENT_FLAT_MATERIAL_NAME)
    return geom_names


def _build_humanoid_wrapper(
    humanoid_root: ET.Element,
    prefix: str,
    wrapper_name: str,
    root_joint_name: str,
    spawn_pos: tuple[float, float, float],
    include_cameras: bool,
    camera_height: float = 1.0,
) -> tuple[str, list[str]]:
    root_body = deepcopy(_find_humanoid_root_body(humanoid_root))

    for child in list(root_body):
        if child.tag == "freejoint":
            root_body.remove(child)

    _remove_tag_recursive(root_body, "camera")
    geom_names = _prefix_named_elements(root_body, prefix)

    wrapper = ET.Element(
        "body",
        {
            "name": wrapper_name,
            "pos": f"{spawn_pos[0]} {spawn_pos[1]} {spawn_pos[2]}",
        },
    )
    ET.SubElement(wrapper, "freejoint", {"name": root_joint_name})

    if include_cameras:
        first_person_cam = {**_FIRST_PERSON_CAMERA, "pos": f"0.15 {1.45 * camera_height:.4f} 0"}
        ET.SubElement(wrapper, "camera", first_person_cam)
        ET.SubElement(wrapper, "camera", dict(_THIRD_PERSON_CAMERA))

    frame = ET.SubElement(
        wrapper,
        "body",
        {
            "name": f"{prefix}frame",
            "euler": _HUMANOID_ROT_EULER,
        },
    )
    frame.append(root_body)

    return ET.tostring(wrapper, encoding="unicode"), geom_names


def build_humanoid_wrapper(
    *,
    prefix: str,
    wrapper_name: str,
    root_joint_name: str,
    spawn_pos: tuple[float, float, float],
    include_cameras: bool,
    camera_height: float = 1.0,
) -> tuple[str, list[str]]:
    humanoid_root = _load_humanoid_root()
    return _build_humanoid_wrapper(
        humanoid_root=humanoid_root,
        prefix=prefix,
        wrapper_name=wrapper_name,
        root_joint_name=root_joint_name,
        spawn_pos=spawn_pos,
        include_cameras=include_cameras,
        camera_height=camera_height,
    )


def build_humanoid_scene(
    mirror_present: bool,
    spawn_pos: tuple[float, float, float],
    camera_height: float = 1.0,
) -> HumanoidSceneFragments:
    humanoid_root = _load_humanoid_root()
    asset_xml = _inject_agent_flat_material(
        _serialize_optional(humanoid_root.find("./asset"))
    )
    default_xml = _serialize_optional(humanoid_root.find("./default"))

    robot_body_xml, robot_geom_names = _build_humanoid_wrapper(
        humanoid_root=humanoid_root,
        prefix="robot_",
        wrapper_name="robot",
        root_joint_name="robot_root",
        spawn_pos=spawn_pos,
        include_cameras=True,
        camera_height=camera_height,
    )

    mirror_body_xml = ""
    mirror_geom_names: list[str] = []
    if mirror_present:
        mirror_body_xml, mirror_geom_names = _build_humanoid_wrapper(
            humanoid_root=humanoid_root,
            prefix="mirror_",
            wrapper_name="mirror_robot",
            root_joint_name="mirror_root",
            spawn_pos=spawn_pos,
            include_cameras=False,
        )

    return HumanoidSceneFragments(
        asset_xml=asset_xml,
        default_xml=default_xml,
        robot_body_xml=robot_body_xml,
        mirror_body_xml=mirror_body_xml,
        robot_geom_names=robot_geom_names,
        mirror_geom_names=mirror_geom_names,
    )
