# 基于 YOLO 的社区目标检测
本项目使用 Ultralytics YOLO 训练三类目标检测模型：社区人员、非社区人员、电动车。

## 交付物

- `runs/train/exp1/weights/best.pt`：验证集表现最好的模型权重
- `runs/train/exp1/`：训练日志、损失/指标曲线、PR/F1 曲线和混淆矩阵
- `runs/eval/exp1_recheck-2/`：独立复核评估图与指标
- `runs/predict/final_results/`：验证图像的推理展示
- `YOLO学习文档.md`：完整的数据、训练、评估和使用说明

## 使用

1. 安装依赖：`pip install ultralytics opencv-python pillow pyyaml`
2. 对新数据集进行质检：`python verify_dataset.py --data community.yaml`
3. 训练：`python train_improved.py --epochs 120 --name exp_improved`
4. 推理：`python predict_improved.py --weights runs/train/exp1/weights/best.pt --source 图片路径`
5. 图形界面：Windows 下双击 `启动图片识别.bat`

## 当前基线结果

`exp1/best.pt` 在 13 张验证图像上的复核结果：Precision 0.699、Recall 0.809、mAP50 0.835、mAP50-95 0.566。数据集规模较小，请参见学习文档中对泛化能力和标注质量的说明。
