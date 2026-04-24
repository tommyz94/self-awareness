from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "mujoco"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_HUMANOID_XML_PATH = _ASSETS_DIR / "humanoid.xml"
_PREFERRED_HUMANOID_XML_PATH = _ASSETS_DIR / "humanoid_real.xml"
_CLOTHES_OBJS_DIR = _REPO_ROOT / "clothes" / "objs"
_LEGACY_AVATAR_OBJS_DIR = _ASSETS_DIR / "objs"
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
_BODY_STYLE_DEFAULT = "default"
_BODY_STYLE_CLOTHES = "clothes"
_SUPPORTED_BODY_STYLES = {_BODY_STYLE_DEFAULT, _BODY_STYLE_CLOTHES}
_AVATAR_TARGET_HEIGHT_M = 1.72
_AVATAR_SEARCH_DIRS = (_CLOTHES_OBJS_DIR, _LEGACY_AVATAR_OBJS_DIR)
_LOD_RE = re.compile(r"_(\d+)k(?:__assimp)?\.obj$", re.IGNORECASE)
_TEXTURE_RES_RE = re.compile(r"_(\d+)k(?:\.[^.]+)?$", re.IGNORECASE)
_TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_TEXTURE_SKIP_TOKENS = (
    "norm",
    "normal",
    "rough",
    "spec",
    "metal",
    "ao",
    "opacity",
    "gloss",
    "height",
    "disp",
    "bump",
)
_TEXTURE_PREFERENCE_TOKENS = (
    "basecolor",
    "albedo",
    "diffuse",
    "color",
    "_dif_",
)


@dataclass(frozen=True)
class HumanoidSceneFragments:
    asset_xml: str
    default_xml: str
    robot_body_xml: str
    mirror_body_xml: str
    robot_geom_names: list[str]
    mirror_geom_names: list[str]


@dataclass(frozen=True)
class AvatarMeshChoice:
    obj_path: Path
    mesh_name: str
    scale: float
    offset: tuple[float, float, float]
    texture_path: Path | None = None
    texture_name: str | None = None
    material_name: str | None = None


def _normalize_body_style(body_style: str) -> str:
    normalized = str(body_style or _BODY_STYLE_DEFAULT).strip().lower()
    if normalized not in _SUPPORTED_BODY_STYLES:
        raise ValueError(
            f"Unsupported body_style={body_style!r}. "
            f"Valid values: {sorted(_SUPPORTED_BODY_STYLES)}"
        )
    return normalized


def _avatar_candidate_sort_key(path: Path) -> tuple[int, int, str]:
    match = _LOD_RE.search(path.name)
    lod = int(match.group(1)) if match else 999_999
    return (lod, len(path.name), path.as_posix().lower())


@lru_cache(maxsize=1)
def _discover_avatar_mesh_candidates() -> tuple[Path, ...]:
    preferred_candidates: list[Path] = []
    for root in _AVATAR_SEARCH_DIRS:
        if not root.exists():
            continue

        grouped: dict[Path, list[Path]] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            current_dir = Path(dirpath)
            if current_dir.name.startswith("."):
                continue
            for filename in filenames:
                if filename.startswith(".") or not filename.lower().endswith(".obj"):
                    continue
                if "__assimp" in filename.lower():
                    continue
                candidate = current_dir / filename
                grouped.setdefault(candidate.parent, []).append(candidate)

        for group in grouped.values():
            preferred_candidates.append(sorted(group, key=_avatar_candidate_sort_key)[0])

    return tuple(sorted(set(preferred_candidates), key=lambda path: path.as_posix().lower()))


@lru_cache(maxsize=32)
def _avatar_bounds(path_str: str) -> tuple[float, float, float, float, float, float]:
    path = Path(path_str)
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    found_vertex = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)
            found_vertex = True
    if not found_vertex:
        raise ValueError(f"Avatar OBJ has no vertices: {path}")
    return (min_x, min_y, min_z, max_x, max_y, max_z)


@lru_cache(maxsize=32)
def _avatar_choice_for_path(path_str: str) -> AvatarMeshChoice:
    path = Path(path_str).resolve()
    min_x, min_y, min_z, max_x, max_y, max_z = _avatar_bounds(path.as_posix())
    span_x = max_x - min_x
    span_y = max_y - min_y
    span_z = max_z - min_z
    reference_height = span_y if span_y > 1e-6 else max(span_x, span_z, 1.0)
    scale = _AVATAR_TARGET_HEIGHT_M / reference_height
    center_x = (min_x + max_x) / 2.0
    center_z = (min_z + max_z) / 2.0
    offset = (
        -center_x * scale,
        -min_y * scale,
        -center_z * scale,
    )
    digest = hashlib.sha1(path.as_posix().encode("utf-8")).hexdigest()[:10]
    mesh_name = f"avatar_mesh_{digest}"
    texture_path = _resolve_avatar_texture_asset(path)
    texture_name = f"avatar_tex_{digest}" if texture_path is not None else None
    material_name = f"avatar_mat_{digest}" if texture_path is not None else None
    return AvatarMeshChoice(
        obj_path=path,
        mesh_name=mesh_name,
        scale=scale,
        offset=offset,
        texture_path=texture_path,
        texture_name=texture_name,
        material_name=material_name,
    )


def sample_avatar_mesh_choice() -> AvatarMeshChoice | None:
    candidates = _discover_avatar_mesh_candidates()
    if not candidates:
        return None
    return _avatar_choice_for_path(random.choice(candidates).as_posix())


def _avatar_mesh_asset_entry(choice: AvatarMeshChoice) -> str:
    scale = f"{choice.scale:.8f} {choice.scale:.8f} {choice.scale:.8f}"
    return (
        f'<mesh name="{choice.mesh_name}" '
        f'file="{choice.obj_path.as_posix()}" '
        f'scale="{scale}"/>'
    )


def _texture_priority(filename: str) -> int:
    lowered = filename.lower()
    for index, token in enumerate(_TEXTURE_PREFERENCE_TOKENS):
        if token in lowered:
            return index
    return len(_TEXTURE_PREFERENCE_TOKENS)


def _texture_resolution(filename: str) -> int:
    match = _TEXTURE_RES_RE.search(filename.lower())
    if match is None:
        return 99_999
    return int(match.group(1))


def _select_texture_source(directory: Path) -> Path | None:
    if not directory.exists():
        return None

    candidates: list[Path] = []
    for candidate in directory.iterdir():
        if (
            not candidate.is_file()
            or candidate.name.startswith(".")
            or candidate.suffix.lower() not in _TEXTURE_SUFFIXES
        ):
            continue
        lowered = candidate.stem.lower()
        if any(token in lowered for token in _TEXTURE_SKIP_TOKENS):
            continue
        candidates.append(candidate)

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda path: (
            _texture_priority(path.name),
            _texture_resolution(path.name),
            len(path.name),
            path.as_posix().lower(),
        ),
    )[0]


def _convert_texture_to_png(texture_path: Path) -> Path:
    resolved = texture_path.resolve()
    if resolved.suffix.lower() == ".png":
        return resolved

    cache_dir = resolved.parent / ".cache_png"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = resolved.stat()
    digest = hashlib.sha1(
        f"{resolved.as_posix()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    cached_path = cache_dir / f"{resolved.stem}_{digest}.png"
    try:
        from PIL import Image
    except ImportError:
        return resolved

    if cached_path.exists():
        try:
            with Image.open(cached_path) as cached_image:
                cached_image.verify()
            return cached_path.resolve()
        except Exception:
            cached_path.unlink(missing_ok=True)

    temp_path = cache_dir / f"{resolved.stem}_{digest}.tmp.png"
    with Image.open(resolved) as image:
        image.convert("RGBA").save(temp_path, format="PNG")
    os.replace(temp_path, cached_path)
    return cached_path.resolve()


@lru_cache(maxsize=32)
def _resolve_avatar_texture_asset(obj_path: Path) -> Path | None:
    for directory in (obj_path.parent / "tex", obj_path.parent):
        source = _select_texture_source(directory)
        if source is not None:
            return _convert_texture_to_png(source)
    return None


def _avatar_asset_xml(choice: AvatarMeshChoice) -> str:
    entries = [_avatar_mesh_asset_entry(choice), _AGENT_FLAT_MATERIAL_XML]
    if (
        choice.texture_path is not None
        and choice.texture_name is not None
        and choice.material_name is not None
    ):
        entries.append(
            f'<texture name="{choice.texture_name}" '
            f'type="2d" file="{choice.texture_path.as_posix()}"/>'
        )
        entries.append(
            f'<material name="{choice.material_name}" '
            f'texture="{choice.texture_name}" '
            'texuniform="true" '
            'specular="0.05" '
            'shininess="0.03"/>'
        )
    return f"<asset>{''.join(entries)}</asset>"


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


def _build_avatar_wrapper(
    *,
    prefix: str,
    wrapper_name: str,
    root_joint_name: str,
    spawn_pos: tuple[float, float, float],
    include_cameras: bool,
    camera_height: float,
    avatar_choice: AvatarMeshChoice,
) -> tuple[str, list[str]]:
    wrapper = ET.Element(
        "body",
        {
            "name": wrapper_name,
            "pos": f"{spawn_pos[0]} {spawn_pos[1]} {spawn_pos[2]}",
        },
    )
    ET.SubElement(wrapper, "freejoint", {"name": root_joint_name})
    ET.SubElement(
        wrapper,
        "inertial",
        {
            "pos": "0 0.9 0",
            "mass": "12.0",
            "diaginertia": "0.25 0.18 0.08",
        },
    )

    if include_cameras:
        first_person_cam = {**_FIRST_PERSON_CAMERA, "pos": f"0.15 {1.45 * camera_height:.4f} 0"}
        ET.SubElement(wrapper, "camera", first_person_cam)
        ET.SubElement(wrapper, "camera", dict(_THIRD_PERSON_CAMERA))

    ox, oy, oz = avatar_choice.offset
    geom_name = f"{prefix}avatar_vis"
    material_name = avatar_choice.material_name or _AGENT_FLAT_MATERIAL_NAME
    ET.SubElement(
        wrapper,
        "geom",
        {
            "name": geom_name,
            "type": "mesh",
            "mesh": avatar_choice.mesh_name,
            "material": material_name,
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "2",
            "pos": f"{ox:.6f} {oy:.6f} {oz:.6f}",
        },
    )
    return ET.tostring(wrapper, encoding="unicode"), [geom_name]


def build_humanoid_wrapper(
    *,
    prefix: str,
    wrapper_name: str,
    root_joint_name: str,
    spawn_pos: tuple[float, float, float],
    include_cameras: bool,
    camera_height: float = 1.0,
    body_style: str = _BODY_STYLE_DEFAULT,
    avatar_choice: AvatarMeshChoice | None = None,
) -> tuple[str, list[str]]:
    normalized_body_style = _normalize_body_style(body_style)
    if normalized_body_style == _BODY_STYLE_CLOTHES:
        choice = avatar_choice or sample_avatar_mesh_choice()
        if choice is not None:
            return _build_avatar_wrapper(
                prefix=prefix,
                wrapper_name=wrapper_name,
                root_joint_name=root_joint_name,
                spawn_pos=spawn_pos,
                include_cameras=include_cameras,
                camera_height=camera_height,
                avatar_choice=choice,
            )

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
    body_style: str = _BODY_STYLE_DEFAULT,
    avatar_choice: AvatarMeshChoice | None = None,
) -> HumanoidSceneFragments:
    normalized_body_style = _normalize_body_style(body_style)
    if normalized_body_style == _BODY_STYLE_CLOTHES:
        choice = avatar_choice or sample_avatar_mesh_choice()
        if choice is not None:
            asset_xml = _avatar_asset_xml(choice)
            robot_body_xml, robot_geom_names = _build_avatar_wrapper(
                prefix="robot_",
                wrapper_name="robot",
                root_joint_name="robot_root",
                spawn_pos=spawn_pos,
                include_cameras=True,
                camera_height=camera_height,
                avatar_choice=choice,
            )
            mirror_body_xml = ""
            mirror_geom_names: list[str] = []
            if mirror_present:
                mirror_body_xml, mirror_geom_names = _build_avatar_wrapper(
                    prefix="mirror_",
                    wrapper_name="mirror_robot",
                    root_joint_name="mirror_root",
                    spawn_pos=spawn_pos,
                    include_cameras=False,
                    camera_height=camera_height,
                    avatar_choice=choice,
                )
            return HumanoidSceneFragments(
                asset_xml=asset_xml,
                default_xml="",
                robot_body_xml=robot_body_xml,
                mirror_body_xml=mirror_body_xml,
                robot_geom_names=robot_geom_names,
                mirror_geom_names=mirror_geom_names,
            )

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
