"""
自动预标注工具（只生成待复审文件，不覆盖人工标注）。

COCO 预训练模型不理解「社区人员/非社区人员」的业务定义，因此 person
只能作为 class 0 候选框，绝不能自动推断为 class 1。导出后必须逐张审核、更正类别与漏标，再写入 labels/ 目录。
"""
import os
from ultralytics import YOLO

# 尝试加载模型
model_path = r"D:\C2\yolo26n.pt"
if not os.path.exists(model_path):
    model_path = "yolov8n.pt"

print(f"加载模型: {model_path}")
model = YOLO(model_path)
print("模型加载成功!")
print(f"模型类别: {model.names}")

# COCO -> 我们的类别
coco_map = {0: 0, 1: 2, 3: 2}  # person->0, bicycle->2, motorcycle->2

def annotate_dir(img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    total_p = 0
    total_s = 0
    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        results = model(img_path, conf=0.25, verbose=False)
        
        lines = []
        for r in results:
            for box in r.boxes:
                cid = int(box.cls[0])
                if cid in coco_map:
                    x, y, w, h = box.xywhn[0].tolist()
                    lines.append(f"{coco_map[cid]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        
        # 写入 labels_auto，保护已复审的真实标注。
        txt_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
        
        np = sum(1 for l in lines if l.startswith("0 "))
        ns = sum(1 for l in lines if l.startswith("2 "))
        total_p += np
        total_s += ns
        print(f"  {img_name}: {np}人 {ns}车")
    
    print(f"  小计: {total_p}人 {total_s}车")

print("\n标注训练集...")
annotate_dir(r"D:\C2\datasets\community\images\train", r"D:\C2\datasets\community\labels_auto\train")

print("\n标注验证集...")
annotate_dir(r"D:\C2\datasets\community\images\val", r"D:\C2\datasets\community\labels_auto\val")

print("\n预标注完成!")
