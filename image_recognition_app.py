"""社区目标检测桌面工具：选择一张图片，使用 YOLO 权重完成识别并保存结果。"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "train" / "exp1" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "runs" / "app_results"
IMAGE_TYPES = [("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")]


class RecognitionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("社区目标检测 - 图片识别")
        self.geometry("1160x800")
        self.minsize(900, 650)
        self.model: YOLO | None = None
        self.model_path = tk.StringVar(value=str(DEFAULT_WEIGHTS))
        self.image_path: Path | None = None
        self.result_image: Image.Image | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        ttk.Label(controls, text="模型权重：").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.model_path, width=82).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(controls, text="选择权重", command=self.choose_weights).grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="选择图片并识别", command=self.choose_image).grid(row=0, column=3, padx=4)
        controls.columnconfigure(1, weight=1)

        options = ttk.Frame(self, padding=(12, 0, 12, 8))
        options.pack(fill="x")
        ttk.Label(options, text="置信度阈值：").pack(side="left")
        self.confidence = tk.DoubleVar(value=0.25)
        ttk.Spinbox(options, from_=0.05, to=0.95, increment=0.05, textvariable=self.confidence, width=6).pack(side="left")
        ttk.Button(options, text="保存当前结果", command=self.save_result).pack(side="right")

        self.status = tk.StringVar(value="请选择图片开始识别。")
        ttk.Label(self, textvariable=self.status, padding=(12, 0, 12, 6)).pack(fill="x")

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        preview_frame = ttk.Labelframe(body, text="识别结果", padding=8)
        info_frame = ttk.Labelframe(body, text="检测明细", padding=10)
        body.add(preview_frame, weight=4)
        body.add(info_frame, weight=1)

        self.canvas = tk.Label(preview_frame, anchor="center", bg="#1f2937", fg="white", text="识别结果将在这里显示")
        self.canvas.pack(fill="both", expand=True)
        self.details = tk.Text(info_frame, width=34, wrap="word", state="disabled", font=("Microsoft YaHei UI", 10))
        self.details.pack(fill="both", expand=True)

    def choose_weights(self) -> None:
        selected = filedialog.askopenfilename(title="选择 YOLO 权重", initialdir=str(ROOT), filetypes=[("PyTorch 权重", "*.pt")])
        if selected:
            self.model_path.set(selected)
            self.model = None
            self.status.set("已更换权重；下次识别时将重新加载模型。")

    def choose_image(self) -> None:
        selected = filedialog.askopenfilename(title="选择待识别图片", initialdir=str(ROOT), filetypes=IMAGE_TYPES)
        if not selected:
            return
        self.image_path = Path(selected)
        self.run_inference()

    def get_model(self) -> YOLO:
        path = Path(self.model_path.get()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"未找到权重文件：{path}")
        if self.model is None:
            self.status.set("正在加载模型，请稍候…")
            self.update_idletasks()
            self.model = YOLO(str(path))
        return self.model

    def run_inference(self) -> None:
        if self.image_path is None:
            return
        try:
            conf = float(self.confidence.get())
            if not 0 < conf < 1:
                raise ValueError("置信度阈值必须在 0 到 1 之间")
            self.status.set("正在识别，请稍候…")
            self.update_idletasks()
            result = self.get_model().predict(source=str(self.image_path), conf=conf, verbose=False)[0]
            plotted = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            self.result_image = Image.fromarray(plotted)
            self.show_image(self.result_image)
            self.show_details(result)
            self.status.set(f"识别完成：{self.image_path.name}，共检测到 {len(result.boxes)} 个目标。")
        except Exception as exc:
            self.status.set("识别失败。")
            messagebox.showerror("识别失败", str(exc))

    def show_image(self, image: Image.Image) -> None:
        # 在保持比例的前提下适配当前预览区；窗口尚未完成布局时使用保守尺寸。
        max_w = max(self.canvas.winfo_width() - 16, 640)
        max_h = max(self.canvas.winfo_height() - 16, 480)
        shown = image.copy()
        shown.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview = ImageTk.PhotoImage(shown)
        self.canvas.configure(image=self.preview, text="")

    def show_details(self, result) -> None:
        labels = result.names
        counts = Counter(int(box.cls[0]) for box in result.boxes)
        lines = ["类别统计"]
        for class_id, label in labels.items():
            lines.append(f"{label}: {counts[class_id]}")
        lines.append("\n逐个目标")
        if not len(result.boxes):
            lines.append("未发现置信度高于阈值的目标。")
        else:
            for index, box in enumerate(result.boxes, 1):
                label = labels[int(box.cls[0])]
                score = float(box.conf[0])
                lines.append(f"{index}. {label}\n   置信度：{score:.1%}")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")

    def save_result(self) -> None:
        if self.result_image is None or self.image_path is None:
            messagebox.showinfo("暂无结果", "请先选择一张图片并完成识别。")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIR / f"{self.image_path.stem}_detected.jpg"
        self.result_image.convert("RGB").save(output, quality=95)
        self.status.set(f"结果已保存：{output}")
        messagebox.showinfo("保存成功", f"结果已保存到：\n{output}")


if __name__ == "__main__":
    try:
        RecognitionApp().mainloop()
    except Exception as error:
        messagebox.showerror("程序启动失败", str(error))
        sys.exit(1)
