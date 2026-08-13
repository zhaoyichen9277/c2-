# 社区人员、非社区人员与电动车目标检测

## 1. 任务与类别定义

使用 Ultralytics YOLO 训练三类目标检测模型：`community_person`、`non_community_person` 和 `electric_scooter`。
类别 0 与 1 是业务类别，不是通用视觉类别：必须在项目开始前写清判定规则（例如制服、工牌、出入区域和行为特征），并由人工按规则标注。COCO 的 `person` 不能被自动分为类 0 或类 1。

## 2. 数据集现状与风险

| 分割 | 图片数 | 社区人员 | 非社区人员 | 电动车 |
|---|---:|---:|---:|---:|
| train | 55 | 219 | 52 | 101 |
| val | 13 | 42 | 13 | 23 |

标注文件的坐标和类别 ID 已通过基础格式检查，但数据量很小，尤其非社区人员只有 52 个训练实例。现有单一验证集的分数不应视为模型上线效果。建议另留 10%–20% 无重复场景的测试集，或做 5 折交叉验证。

## 3. 训练前质检

```powershell
cd D:\C2
.\.venv\Scripts\python.exe verify_dataset.py --data community.yaml
```
该脚本会检查图片/标注对应、YOLO 坐标范围、类别 ID、空标注和 train/val 间的重复图片，并生成 `runs/audit/dataset_audit.json`。
自动预标注只能用作候选框生成：

```powershell
.\.venv\Scripts\python.exe auto_annotate.py
```

输出将保存在 `labels_auto/`，不会覆盖人工标注。需逐张复审边界框、漏标与类别 0/1。对很远、被遮挡或难以判定的对象，应在标注规范中统一排除或标注准则。

## 4. 改进训练方案

```powershell
.\.venv\Scripts\python.exe train_improved.py --epochs 120 --name exp_improved
```

脚本使用 COCO 预训练权重、固定随机种子、AdamW、Cosine 学习率和适度的几何/颜色增强。有 CUDA 时自动选用 GPU，否则使用 CPU。`patience=35` 与 `close_mosaic=15` 防止在小数据上过早结束，也让最后阶段更接近真实分布。
如果非社区人员的召回仍偏低，先补标、增加该类的独立场景、服饰和光照样本，再考虑更大模型 `yolo26s.pt`。不建议仅靠重复图片或把类别 0 的伪标注改为类 1 来「平衡」数据。

## 5. 评估、曲线与推理

```powershell
.\.venv\Scripts\python.exe evaluate_model.py --weights runs/train/exp_improved/weights/best.pt
.\.venv\Scripts\python.exe plot_results.py --csv runs/train/exp_improved/results.csv
.\.venv\Scripts\python.exe predict_improved.py --weights runs/train/exp_improved/weights/best.pt --source datasets/community/images/val --conf 0.25
```

`evaluate_model.py` 会导出每类 Precision、Recall、mAP50 和 mAP50-95 到 `metrics.json`，不使用写死的类别分数或最佳 epoch。`plot_results.py` 从实际 CSV 动态获取最佳 epoch。推理与验证时应固定一个阈值并报告它；可结合 F1 曲线选择阈值，而不能只挑选对自己有利的图片。

## 6. 已有实验的正确解读

现有 `exp1` 在 40 个 epoch 中的最佳验证 mAP50-95 为 0.56651（epoch 34）。该结果可作为基线，但由于验证集很小且类别含业务语义，不应直接宣称模型已具有充分的实际泛化能力。报告中应同时提供数据分割、每类实例数、混淆矩阵、PR/F1 曲线以及未参与超参数选择的测试结果。

对提供的 `bus.jpg` 用现有 `exp1/best.pt` 在 `conf=0.15` 下推理未产生任何检测框。这个结果说明不应只靠验证集分数判断泛化能力：请补充不同摄像头、角度、光照、背景的目标场景样本，且将完全独立的场景作为测试集。

## 7. 交付清单

- 权重：`runs/train/exp_improved/weights/best.pt`
- 日志与曲线：`results.csv`、`results.png`、`results_plot.png`、`confusion_matrix.png`、`PR_curve.png`、`F1_curve.png`
- 评估汇总：`runs/eval/final_eval/metrics.json`
- 推理示例：`runs/predict/improved_results/`
- 本文档与 `community.yaml`、`verify_dataset.py`、`train_improved.py`、`evaluate_model.py`、`predict_improved.py`

## 8. 桌面图片识别工具

双击 `启动图片识别.bat`，或在 PowerShell 中执行：

```powershell
cd D:\C2
.\.venv\Scripts\python.exe image_recognition_app.py
```

程序默认加载 `runs/train/exp1/weights/best.pt`。点击“选择图片并识别”，设定置信度阈值（建议先使用 0.25），程序将在界面展示边界框、类别、置信度和每类数量。点击“保存当前结果”后，结果图会保存到 `runs/app_results/`。

当重新训练出 `exp_improved` 后，在界面点击“选择权重”，选取 `runs/train/exp_improved/weights/best.pt` 即可切换到新模型，无需改代码。
