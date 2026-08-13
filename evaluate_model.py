"""
输出总体与每类评估指标，避免把类别指标写死在脚本中。
用法：python evaluate_model.py --weights runs/train/exp_improved/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/exp1/weights/best.pt")
    parser.add_argument("--data", default="community.yaml")
    parser.add_argument("--name", default="final_eval")
    args = parser.parse_args()

    model = YOLO(str(resolve(args.weights)))
    metrics = model.val(data=str(resolve(args.data)), split="val", plots=True,
                        project=str(ROOT / "runs/eval"), name=args.name, exist_ok=False)
    per_class = {}
    for index, name in model.names.items():
        per_class[name] = {
            "precision": round(float(metrics.box.p[index]), 6),
            "recall": round(float(metrics.box.r[index]), 6),
            "map50": round(float(metrics.box.ap50[index]), 6),
            "map50_95": round(float(metrics.box.ap[index]), 6),
        }
    summary = {"overall": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                           "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
               "per_class": per_class}
    # Ultralytics 在同名目录存在时会自动增加后缀，因此从返回的 metrics 读取实际输出目录。
    output = Path(metrics.save_dir) / "metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"评估结果已保存：{output}")


if __name__ == "__main__":
    main()
