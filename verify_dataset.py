"""
训练前数据集质检：检查 YOLO 标注格式、类别分布、缺失标注与集间泄漏。
用法：python verify_dataset.py --data community.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_split(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_label(label: Path, nc: int) -> tuple[Counter, list[str]]:
    counts, errors = Counter(), []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError("字段数不是 5")
            cls = int(fields[0])
            values = [float(x) for x in fields[1:]]
            if not 0 <= cls < nc:
                raise ValueError(f"类别 ID {cls} 超出 0..{nc - 1}")
            if not all(0.0 <= x <= 1.0 for x in values):
                raise ValueError("归一化坐标不在 [0, 1]")
            if values[2] <= 0 or values[3] <= 0:
                raise ValueError("框宽高必须大于 0")
            counts[cls] += 1
        except (ValueError, IndexError) as exc:
            errors.append(f"{label}:{line_no}: {exc}")
    return counts, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="community.yaml")
    parser.add_argument("--output", default="runs/audit/dataset_audit.json")
    args = parser.parse_args()

    yaml_path = Path(args.data).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    dataset_root = resolve_split(yaml_path.parent, config["path"])
    names = config["names"]
    nc = len(names)
    report: dict[str, object] = {"data": str(yaml_path), "classes": names, "splits": {}}
    seen_hashes: dict[str, str] = {}
    all_errors: list[str] = []

    for split in ("train", "val", "test"):
        if split not in config:
            continue
        image_dir = resolve_split(dataset_root, config[split])
        label_dir = Path(str(image_dir).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/"))
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        counts, missing, empty, duplicates = Counter(), [], [], []
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                missing.append(image.name)
            else:
                if not label.read_text(encoding="utf-8").strip():
                    empty.append(image.name)
                current_counts, errors = validate_label(label, nc)
                counts.update(current_counts)
                all_errors.extend(errors)
            digest = file_hash(image)
            if digest in seen_hashes:
                duplicates.append({"image": image.name, "same_as": seen_hashes[digest]})
            else:
                seen_hashes[digest] = f"{split}/{image.name}"
        report["splits"][split] = {
            "images": len(images), "objects_by_class": {str(k): counts[k] for k in range(nc)},
            "missing_labels": missing, "empty_labels": empty, "duplicate_images": duplicates,
        }

    report["label_errors"] = all_errors
    output = Path(args.output)
    if not output.is_absolute():
        output = yaml_path.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n审计报告已保存：{output}")
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
