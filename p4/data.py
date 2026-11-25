"""Prepare RGB/split manifests; reference cameras are evaluation-only.

Sources: COLMAP ``images`` + ``sparse/0`` (text or binary), NeRF/nerfstudio
``transforms*.json``, course/S3PO ``rgb`` + ``cameras.json``, or an RGB ``images`` directory. Pose-free RGB sources can
be initialized/trained, but cannot provide pose or held-out rendering evaluation.
An official split is never inferred from frame numbering without an explicit
``test_every`` fallback, whose provenance is recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12), 11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
COURSE_CAMERA_SOURCE = (
    "https://github.com/3DAgentWorld/S3PO-GS/blob/"
    "d3a5c0926e5c5bceac362e5d0646e60575cbcd0a/utils/dataset.py"
)


def natural_key(value):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r"(\d+)", str(value))]


def _read(stream, fmt):
    fmt = "<" + fmt
    size = struct.calcsize(fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary model")
    return struct.unpack(fmt, data)


def qvec_to_rotation(qvec):
    q = np.asarray(qvec, dtype=np.float64)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])


def _camera(model, width, height, params):
    p = np.asarray(params, dtype=np.float64)
    simple = model.startswith("SIMPLE_") or model in {"RADIAL", "RADIAL_FISHEYE"}
    if simple:
        fx, cx, cy = p[:3]
        fy, extra = fx, p[3:]
    else:
        fx, fy, cx, cy = p[:4]
        extra = p[4:]
    return {"model": model, "width": int(width), "height": int(height),
            "K": np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]),
            "distortion": extra}


def read_colmap_model(model_dir):
    """Return cameras and registered image records from COLMAP txt/bin.

    Image record values contain ``camera_id`` and OpenCV ``c2w``. No points are
    loaded: source model geometry is reference information, not an initializer.
    """
    model_dir = Path(model_dir)
    cameras, images = {}, {}
    if (model_dir / "cameras.bin").exists():
        with (model_dir / "cameras.bin").open("rb") as stream:
            for _ in range(_read(stream, "Q")[0]):
                cid, mid, width, height = _read(stream, "iiQQ")
                if mid not in CAMERA_MODELS:
                    raise ValueError(f"Unknown COLMAP camera model id {mid}")
                name, count = CAMERA_MODELS[mid]
                cameras[cid] = _camera(name, width, height, _read(stream, "d" * count))
        with (model_dir / "images.bin").open("rb") as stream:
            for _ in range(_read(stream, "Q")[0]):
                record = _read(stream, "idddddddi")
                name = bytearray()
                while True:
                    byte = stream.read(1)
                    if not byte:
                        raise ValueError("Truncated COLMAP image name")
                    if byte == b"\0":
                        break
                    name.extend(byte)
                c2w = np.eye(4)
                rot = qvec_to_rotation(record[1:5])
                c2w[:3, :3] = rot.T
                c2w[:3, 3] = -rot.T @ np.asarray(record[5:8])
                images[name.decode("utf-8")] = {"camera_id": record[8], "c2w": c2w}
                count = _read(stream, "Q")[0]
                if len(stream.read(count * 24)) != count * 24:
                    raise ValueError("Truncated COLMAP observations")
    elif (model_dir / "cameras.txt").exists():
        for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            cameras[int(fields[0])] = _camera(fields[1], int(fields[2]), int(fields[3]), fields[4:])
        lines = (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) < 10:
                raise ValueError("Invalid COLMAP images.txt pose line")
            rot = qvec_to_rotation([float(v) for v in fields[1:5]])
            c2w = np.eye(4)
            c2w[:3, :3] = rot.T
            c2w[:3, 3] = -rot.T @ np.asarray(fields[5:8], dtype=float)
            images[fields[9]] = {"camera_id": int(fields[8]), "c2w": c2w}
            i += 1  # The next physical line contains observations (possibly empty).
    else:
        raise FileNotFoundError(f"No COLMAP cameras.txt/bin in {model_dir}")
    for name, record in images.items():
        if record["camera_id"] not in cameras:
            raise ValueError(f"Missing camera for {name}")
    return cameras, images


def _safe_relative(path):
    p = Path(str(path).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Image path must remain within dataset: {path}")
    return p


def _image_path(source, name):
    rel = _safe_relative(name)
    candidates = [source / rel, source / "images" / rel]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if not candidate.suffix:
            found = [candidate.with_suffix(ext) for ext in IMAGE_EXTENSIONS
                     if candidate.with_suffix(ext).is_file()]
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                raise ValueError(f"Ambiguous extension for {name}")
    raise FileNotFoundError(f"Source image not found: {name}")


def _course_records(source):
    """Read the inspected mandatory-data archive's S3PO-compatible schema.

    The upstream dl3dvParser uses scipy Rotation.from_quat (xyzw), constructs
    [R | cam_trans] as c2w, then inverts it for its w2c consumer. Its dl3dv/2
    calibration confirms normalized intrinsics multiply their corresponding
    image dimensions (fx=.79706059*256=204.04751 pixels). No OpenGL flip applies.
    Camera metadata is isolated in reference.npz; initializers see only RGB.
    """
    image_dir = source / "rgb" if (source / "rgb").is_dir() else source / "images"
    poses = json.loads((source / "cameras.json").read_text(encoding="utf-8"))
    if not isinstance(poses, list) or not poses:
        raise ValueError("Course cameras.json must be a nonempty list of camera records")
    shared = {}
    if (source / "intrinsics.json").exists():
        shared = json.loads((source / "intrinsics.json").read_text(encoding="utf-8"))
        if not isinstance(shared, dict):
            raise ValueError("Course intrinsics.json must be an object")
    images = {p.relative_to(image_dir).as_posix(): p for p in image_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
              and not any(part.startswith(".") for part in p.relative_to(image_dir).parts)}
    records, refs = {}, {}
    for record in poses:
        name = _safe_relative(record["image_name"]).as_posix()
        if name in records:
            raise ValueError(f"Duplicate course camera image_name: {name}")
        if name not in images:
            raise ValueError(f"Course camera image missing: {name}; source archive may be incomplete")
        image = images[name]
        with Image.open(image) as im:
            width, height = im.size
        calibration = {**shared, **record}
        try:
            fx, fy, cx, cy = [float(calibration[key]) for key in ("fx", "fy", "cx", "cy")]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Course normalized intrinsics missing: {name}") from exc
        if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
            raise ValueError(f"Invalid course normalized intrinsics: {name}")
        quaternion = np.asarray(record["cam_quat"], dtype=np.float64)
        translation = np.asarray(record["cam_trans"], dtype=np.float64)
        if quaternion.shape != (4,) or translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"Invalid course camera pose: {name}")
        c2w = np.eye(4)
        c2w[:3, :3] = qvec_to_rotation(quaternion[[3, 0, 1, 2]])
        c2w[:3, 3] = translation
        records[name] = {"path": image, "split": None}
        refs[name] = {"c2w": c2w,
                      "K": np.array([[fx * width, 0, cx * width], [0, fy * height, cy * height], [0, 0, 1.]]),
                      "width": width, "height": height, "model": "PINHOLE", "distortion": np.zeros(0)}
    extras = set(images) - records.keys()
    if extras:
        raise ValueError(f"RGB images absent from course cameras.json: {sorted(extras)[:5]}")
    return records, refs, None


def _waymo_records(source):
    """Read S3PO Waymo FRONT RGB, per-frame calibration, and camera gt poses.

    Upstream WaymoParser treats gt/*.txt as camera c2w. Tr_velo_to_cam in
    calib is not applied a second time; gt_old is not the chosen reference.
    Nonzero Brown-Conrady distortion is removed from every canonical RGB before
    resize by the same preprocessing path as other calibrated sources.
    """
    image_dir = source / "rgb"
    image_files = {p.relative_to(image_dir).as_posix(): p for p in image_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                   and not any(part.startswith(".") for part in p.relative_to(image_dir).parts)}
    if not image_files:
        raise ValueError("Waymo RGB directory is empty")
    records, refs = {}, {}
    for name, image in image_files.items():
        stem = Path(name).with_suffix("")
        pose_path = source / "gt" / stem.with_suffix(".txt")
        calibration_path = source / "calib" / stem.with_suffix(".txt")
        if not pose_path.is_file() or not calibration_path.is_file():
            raise ValueError(f"Waymo RGB lacks matching gt/calib: {name}; source archive may be incomplete")
        pose = np.loadtxt(pose_path, dtype=np.float64)
        if pose.size == 16:
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"Invalid Waymo gt camera pose: {name}")
        R = pose[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1, atol=1e-4):
            raise ValueError(f"Waymo gt camera rotation is not SO(3): {name}")
        text = calibration_path.read_text(encoding="utf-8")
        values = {key: float(value) for key, value in re.findall(
            r"\b(fx|fy|cx|cy|k1|k2|k3|p1|p2)\s*:\s*([-+0-9.eE]+)", text)}
        if any(key not in values for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")):
            raise ValueError(f"Incomplete Waymo calibration: {name}")
        if not np.isfinite(list(values.values())).all() or min(values["fx"], values["fy"]) <= 0:
            raise ValueError(f"Invalid Waymo calibration: {name}")
        with Image.open(image) as im:
            width, height = im.size
        K = np.array([[values["fx"], 0, values["cx"]], [0, values["fy"], values["cy"]], [0, 0, 1.]])
        if not (0 <= K[0, 2] <= width and 0 <= K[1, 2] <= height):
            raise ValueError(f"Waymo intrinsics do not match RGB dimensions: {name}")
        records[name] = {"path": image, "split": None}
        refs[name] = {"c2w": pose, "K": K, "width": width, "height": height,
                      "model": "OPENCV", "distortion": np.asarray([values[k] for k in ("k1", "k2", "p1", "p2", "k3")])}
    # Do not quietly interpret a truncated RGB extraction as the complete scene.
    expected_stems = {Path(name).with_suffix("").as_posix() for name in image_files}
    for group in ("gt", "calib"):
        actual_stems = {p.relative_to(source / group).with_suffix("").as_posix()
                        for p in (source / group).rglob("*.txt")
                        if not any(part.startswith(".") for part in p.relative_to(source / group).parts)}
        if actual_stems != expected_stems:
            raise ValueError(f"Waymo {group}/RGB frame coverage differs; source archive may be incomplete")
    return records, refs, None


def _source_records(source):
    if ((source / "rgb").is_dir() or (source / "images").is_dir()) and (source / "cameras.json").is_file():
        return _course_records(source)
    if all((source / name).is_dir() for name in ("rgb", "gt", "calib")):
        return _waymo_records(source)
    transforms = [source / "transforms.json"]
    explicit_split = None
    if not transforms[0].exists():
        transforms = [p for p in (source / "transforms_train.json", source / "transforms_test.json") if p.exists()]
    records, refs = {}, {}
    if transforms:
        for path in transforms:
            data = json.loads(path.read_text(encoding="utf-8"))
            group = "train" if path.stem == "transforms_train" else "test" if path.stem == "transforms_test" else None
            for frame in data["frames"]:
                image = _image_path(source, frame["file_path"])
                rel = image.relative_to(source / "images") if image.is_relative_to(source / "images") else image.relative_to(source)
                name = rel.as_posix()
                if name in records:
                    raise ValueError(f"Duplicate transforms frame: {name}")
                with Image.open(image) as im:
                    w, h = im.size
                meta = {**data, **frame}
                rw, rh = int(meta.get("w", w)), int(meta.get("h", h))
                fx = meta.get("fl_x")
                if fx is None:
                    if "camera_angle_x" not in meta:
                        raise ValueError(f"Missing focal length: {name}")
                    fx = 0.5 * rw / math.tan(float(meta["camera_angle_x"]) / 2)
                fy = float(meta.get("fl_y", fx))
                K = np.array([[float(fx), 0, float(meta.get("cx", rw/2))],
                              [0, fy, float(meta.get("cy", rh/2))], [0, 0, 1.]])
                # NeRF transforms convention is OpenGL; no ambiguous implicit axis guessing.
                pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
                if pose.shape == (3, 4):
                    pose = np.vstack([pose, [0, 0, 0, 1]])
                convention = data.get("camera_convention", "opengl").lower()
                if convention not in {"opengl", "opencv"}:
                    raise ValueError(f"Unsupported camera_convention {convention}")
                if convention == "opengl":
                    pose = pose @ np.diag([1., -1., -1., 1.])
                # OpenCV's rational model accepts 8 coefficients, not a six-
                # coefficient array. Nerfstudio k4 is the first denominator term.
                distortion = np.array([float(meta.get(k, 0)) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
                model = str(meta.get("camera_model", "OPENCV")).upper()
                if model not in {"OPENCV", "PINHOLE", "SIMPLE_PINHOLE"}:
                    raise ValueError(f"Unsupported transforms camera_model: {model}")
                records[name] = {"path": image, "split": group}
                refs[name] = {"c2w": pose, "K": K, "width": rw, "height": rh,
                              "model": model, "distortion": distortion}
            if path.name == "transforms.json" and ("train_filenames" in data or "test_filenames" in data):
                explicit_split = {"train": data.get("train_filenames", []), "test": data.get("test_filenames", [])}
        if all(record["split"] for record in records.values()):
            explicit_split = {split: [name for name, rec in records.items() if rec["split"] == split]
                              for split in ("train", "test")}
        return records, refs, explicit_split
    image_dir = source / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected an images directory or transforms*.json under {source}")
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            records[p.relative_to(image_dir).as_posix()] = {"path": p, "split": None}
    model = next((p for p in (source / "sparse/0", source / "sparse", source)
                  if (p / "cameras.bin").exists() or (p / "cameras.txt").exists()), None)
    if model:
        cameras, images = read_colmap_model(model)
        for name, rec in images.items():
            if name in records:
                refs[name] = {**cameras[rec["camera_id"]], "c2w": rec["c2w"]}
    return records, refs, None


def _resolve_split(split, names):
    output = {}
    for group in ("train", "test"):
        if not isinstance(split.get(group), list):
            raise ValueError("Split JSON must contain train and test filename lists")
        for value in split[group]:
            value = str(value).replace("\\", "/").removeprefix("./")
            candidates = {value, value.removeprefix("images/")}
            matches = [n for n in names if n in candidates]
            if not matches:
                # Stem-only frame IDs are accepted only when unambiguous.
                matches = [n for n in names if Path(n).stem == value]
            if len(matches) != 1:
                raise ValueError(f"Split frame missing or ambiguous: {value}")
            name = matches[0]
            if name in output:
                raise ValueError(f"Duplicate or overlapping split frame: {name}")
            output[name] = group
    missing = set(names) - output.keys()
    if missing:
        raise ValueError(f"Split must account for every source image; unlisted: {sorted(missing)[:5]}")
    return output


def _undistort(image, camera):
    distortion = camera["distortion"]
    if not np.any(np.abs(distortion) > 1e-12):
        return image
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Nonzero lens distortion requires opencv-python-headless") from exc
    model = camera["model"]
    if model in {"SIMPLE_RADIAL", "RADIAL"}:
        coeff = np.zeros(5)
        coeff[:len(distortion)] = distortion
    elif model in {"OPENCV", "FULL_OPENCV"}:
        coeff = distortion
    elif model in {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        coeff = np.zeros(4)
        coeff[:len(distortion)] = distortion
        return Image.fromarray(cv2.fisheye.undistortImage(np.asarray(image), camera["K"], coeff,
                                                        Knew=camera["K"], new_size=image.size))
    else:
        raise ValueError(f"Unsupported distortion model: {model}; undistort source explicitly")
    return Image.fromarray(cv2.undistort(np.asarray(image), camera["K"], coeff, None, camera["K"]))


def prepare_dataset(source, output, scene, test_every=None, max_size=960, split_file=None,
                    reference_kind="reference"):
    """Write canonical images, frozen manifest, and isolated reference.npz.

    ``split_file`` is JSON ``{"train": [...], "test": [...]}``. Without an
    official source split, ``test_every`` must explicitly choose the documented
    fallback (indices 0, k, 2k, ... held out). Reference kind must truthfully be
    ``ground_truth`` or ``reference``; no COLMAP pose is automatically called GT.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if not (source / "rgb").is_dir() and all((source / "FRONT" / name).is_dir() for name in ("rgb", "gt", "calib")):
        source = source / "FRONT"
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and prepared output must be separate, non-nested directories")
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared dataset already exists: {output}; choose a new output")
    if reference_kind not in {"ground_truth", "reference"}:
        raise ValueError("reference_kind must be ground_truth or reference")
    if max_size is not None and max_size < 16:
        raise ValueError("max_size must be at least 16")
    records, refs, official = _source_records(source)
    names = sorted(records, key=natural_key)
    if len(names) < 3:
        raise ValueError("Need at least three source images")
    if split_file:
        content = Path(split_file).read_text(encoding="utf-8")
        split = _resolve_split(json.loads(content), names)
        split_source = {"kind": "explicit", "file": str(Path(split_file).resolve()),
                        "sha256": hashlib.sha256(content.encode()).hexdigest()}
    elif official:
        split = _resolve_split(official, names)
        split_source = {"kind": "source_transforms", "note": "train/test split supplied with source"}
    elif test_every is not None:
        if test_every < 2:
            raise ValueError("test_every must be at least 2")
        split = {name: "test" if i % test_every == 0 else "train" for i, name in enumerate(names)}
        split_source = {"kind": "fallback", "test_every": test_every,
                        "rule": "natural filename order; test indices 0,k,2k,...; not an official split"}
    else:
        raise ValueError("No official split found. Supply --split-file or explicitly select --test-every.")
    if not all(group in split.values() for group in ("train", "test")):
        raise ValueError("Split requires nonempty train and test sets")
    frames, reference_names, reference_poses, reference_K = [], [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        record = records[name]
        with Image.open(record["path"]) as source_image:
            if source_image.getexif().get(274, 1) != 1:
                raise ValueError(f"EXIF-rotated image {name}: normalize before providing camera calibration")
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        ow, oh = image.size
        camera = refs.get(name)
        if camera:
            if (camera["width"], camera["height"]) != image.size:
                raise ValueError(f"Source camera dimensions do not match image: {name}")
            image = _undistort(image, camera)
        ratio = min(1., max_size / max(ow, oh)) if max_size else 1.
        width, height = max(1, round(ow * ratio)), max(1, round(oh * ratio))
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        # Preserve complete relative names; PNG prevents different JPEG re-encoding
        # from becoming a hidden experimental variable.
        target_rel = Path("images") / Path(name).with_suffix(Path(name).suffix + ".png")
        target = output / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        frame = {"id": name, "image": target_rel.as_posix(), "split": split[name],
                 "width": width, "height": height,
                 "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        frames.append(frame)
        if camera:
            K = camera["K"].copy()
            K[0, :] *= width / ow
            K[1, :] *= height / oh
            reference_names.append(target_rel.as_posix())
            reference_poses.append(camera["c2w"])
            reference_K.append(K)
    manifest = {"schema_version": 1, "scene": scene, "source": str(source),
                "split_source": split_source, "camera_convention": "opencv_c2w",
                "reference_kind": reference_kind if reference_names else "unavailable",
                "reference_poses": len(reference_names), "max_size": max_size,
                "preprocessing": "RGB PNG; calibrated distortion removed before resize; no crop",
                "frames": frames}
    if ((source / "rgb").is_dir() or (source / "images").is_dir()) and (source / "cameras.json").is_file():
        folder = "rgb" if (source / "rgb").is_dir() else "images"
        manifest.update(source_format=f"course_{folder}_cameras_json",
                        reference_pose_convention="OpenCV c2w; cam_quat=xyzw; cam_trans=camera center",
                        reference_intrinsics="fx,cx multiply width; fy,cy multiply height",
                        reference_convention_source=COURSE_CAMERA_SOURCE)
    elif all((source / name).is_dir() for name in ("rgb", "gt", "calib")):
        manifest.update(source_format="course_waymo_front",
                        reference_pose_convention="OpenCV c2w from gt/*.txt; gt_old and Tr_velo_to_cam unused",
                        reference_intrinsics="per-frame calib/*.txt pixel intrinsics; Brown-Conrady lens distortion removed",
                        reference_convention_source=COURSE_CAMERA_SOURCE)
    # Bind the actual RGB inputs and assignment, independent of workstation
    # absolute paths so a server re-prepare produces the same split identity.
    fingerprint = {"schema_version": 1, "scene": scene, "camera_convention": "opencv_c2w",
                   "max_size": max_size, "preprocessing": manifest["preprocessing"],
                   "frames": frames}
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest["split_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if reference_names:
        np.savez_compressed(output / "reference.npz", names=np.asarray(reference_names),
                            c2w=np.asarray(reference_poses), K=np.asarray(reference_K))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(root):
    root = Path(root)
    path = root if root.name == "manifest.json" else root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    names = [f["image"] for f in manifest["frames"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image names in manifest")
    for frame in manifest["frames"]:
        _safe_relative(frame["image"])
        if frame["split"] not in {"train", "test"}:
            raise ValueError("Invalid manifest split")
    return manifest


def load_scene(path):
    """Load and validate the shared train-only initializer NPZ schema."""
    path = Path(path)
    path = path / "scene.npz" if path.is_dir() else path
    with np.load(path, allow_pickle=False) as array:
        scene = {key: array[key] for key in ("names", "c2w", "K", "points", "colors")}
    count = len(scene["names"])
    if not count or scene["c2w"].shape != (count, 4, 4) or scene["K"].shape != (count, 3, 3):
        raise ValueError("Invalid scene camera shapes")
    if scene["points"].ndim != 2 or scene["points"].shape[1] != 3 or not len(scene["points"]):
        raise ValueError("Scene requires a nonempty M x 3 point cloud")
    if scene["colors"].shape != scene["points"].shape:
        raise ValueError("Point colors must have the same M x 3 shape")
    if len(set(scene["names"].tolist())) != count:
        raise ValueError("Duplicate scene image names")
    if scene["colors"].dtype != np.uint8:
        raise ValueError("Point colors must be uint8 in [0,255]")
    for key in ("c2w", "K", "points"):
        if not np.isfinite(scene[key]).all():
            raise ValueError(f"Nonfinite scene {key}")
    if not np.allclose(scene["c2w"][:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous c2w bottom row")
    rotations = scene["c2w"][:, :3, :3]
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("c2w rotation determinant must be +1")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("c2w rotation must be orthonormal")
    if np.any(scene["K"][:, [0, 1], [0, 1]] <= 0):
        raise ValueError("Focal lengths must be positive")
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--split-file")
    prepare.add_argument("--test-every", type=int)
    prepare.add_argument("--max-size", type=int, default=960)
    prepare.add_argument("--reference-kind", choices=["reference", "ground_truth"], default="reference")
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("command")
    result = prepare_dataset(**kwargs)
    print(json.dumps({"scene": result["scene"], "split_hash": result["split_hash"],
                      "train": sum(f["split"] == "train" for f in result["frames"]),
                      "test": sum(f["split"] == "test" for f in result["frames"]),
                      "reference_kind": result["reference_kind"]}, indent=2))


if __name__ == "__main__":
    main()
