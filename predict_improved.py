"""
使用最佳权重批量推理，保存图片、TXT 检测结果和置信度。"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/exp1/weights/best.pt")
    parser.add_argument("--source", default="datasets/community/images/val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--name", default="improved_results")
    args = parser.parse_args()
    model = YOLO(str(resolve(args.weights)))
    results = model.predict(source=str(resolve(args.source)), conf=args.conf, iou=args.iou,
                            save=True, save_txt=True, save_conf=True,
                            project=str(ROOT / "runs/predict"), name=args.name, exist_ok=False)
    for result in results:
        counts: dict[str, int] = {}
        for box in result.boxes:
            name = model.names[int(box.cls[0])]
            counts[name] = counts.get(name, 0) + 1
        print(f"{Path(result.path).name}: {counts}")


if __name__ == "__main__":
    main()
