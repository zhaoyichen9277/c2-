"""
面向小型数据集的可复现 YOLO 训练脚本。训练前请先运行 verify_dataset.py，
并且必须人工复审所有「社区/非社区人员」标注。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt", help="预训练权重，可改为 yolo26s.pt")
    parser.add_argument("--data", default="community.yaml")
    parser.add_argument("--name", default="exp_improved")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=0, help="0=自动；CPU 默认 8")
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    batch = args.batch if args.batch else (-1 if device != "cpu" else 8)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=batch,
        device=device, workers=2 if device != "cpu" else 0, optimizer="AdamW",
        lr0=0.002, lrf=0.01, cos_lr=True, warmup_epochs=3.0,
        patience=35, close_mosaic=15, mosaic=0.8, mixup=0.05,
        degrees=5.0, translate=0.08, scale=0.35, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        seed=42, deterministic=True, pretrained=True,
        project=str(ROOT / "runs/train"), name=args.name, exist_ok=False,
    )
    metrics = model.val(data=str(data_path), split="val")
    print(f"mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
