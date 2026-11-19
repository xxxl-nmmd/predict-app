# predict_app.py
import os
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# 关键：导入自定义类，让 joblib.load 能解析 pkl 里的类引用
from stacking_model import RobustDataPreprocessor, HyperparameterOptimizer, StackingEnsembleModel

MODEL_DIR = "trained_models"

# ---------- 输入特征定义（与训练一致） ----------
INPUT_GROUPS = {
    "生化组成": [("Pro (%)", "蛋白质 %"), ("Lip (%)", "脂肪 %"), ("CH (%)", "碳水化合物 %")],
    "元素分析": [("C (%)", "C %"), ("H (%)", "H %"), ("O (%)", "O %"),
                 ("N (%)", "N %"), ("S (%)", "S %")],
    "灰分":     [("Ash (%)", "灰分 %")],
    "操作条件": [("Time (min)", "停留时间 min"), ("T (℃)", "温度 ℃"),
                 ("Size (L)", "反应器体积 L"), ("Ratio", "生物质/溶剂比")],
}
SOLVENTS = [("RS", "反应溶剂"), ("SS", "分离溶剂")]
ALL_INPUTS = [c for g in INPUT_GROUPS.values() for c, _ in g] + [c for c, _ in SOLVENTS]


def load_models():
    models = {}
    if not os.path.isdir(MODEL_DIR):
        print(f"警告: 模型目录不存在: {MODEL_DIR}")
        return models
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            try:
                d = joblib.load(os.path.join(MODEL_DIR, f))
                models[d["target_name"]] = d
                print(f"成功加载模型: {f}")
            except Exception as e:
                print(f"加载 {f} 失败: {e}")
    return models


def build_input_df(values: dict, meta: dict) -> pd.DataFrame:
    """values: 原始特征名 -> 数值/字符串；meta: 该目标模型的元信息"""
    row = {}
    for c in meta["feature_order"]:
        row[c] = values.get(c, np.nan)
    df = pd.DataFrame([row])

    # 独热编码（用训练时的类别水平）
    for c in meta["categorical_cols"]:
        levels = meta["category_levels"].get(c, [])
        for lv in levels:
            df[f"{c}_{lv}"] = int(str(values.get(c)) == str(lv))
        df.drop(columns=[c], inplace=True)

    return df


class App:
    def __init__(self, root):
        self.root = root
        root.title("水热碳化液相生物油性质预测软件")
        self.models = load_models()
        self.entries = {}
        self._build_ui()

        if not self.models:
            messagebox.showwarning(
                "无模型",
                f"未在 '{MODEL_DIR}' 目录找到模型文件。\n请先运行训练脚本生成 pkl。"
            )

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        r = 0
        for group, items in INPUT_GROUPS.items():
            ttk.Label(frm, text=group, font=("", 10, "bold")).grid(
                row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
            r += 1
            for col, label in items:
                ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w")
                e = ttk.Entry(frm, width=18)
                e.grid(row=r, column=1, sticky="w")
                self.entries[col] = e
                r += 1

        for col, label in SOLVENTS:
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w")
            cb = ttk.Combobox(frm, width=15, values=self._solvent_levels(col))
            cb.grid(row=r, column=1, sticky="w")
            self.entries[col] = cb
            r += 1

        ttk.Button(frm, text="预测", command=self.predict).grid(
            row=r, column=0, columnspan=2, pady=10)
        r += 1
        self.out = tk.Text(frm, width=60, height=14, state="disabled")
        self.out.grid(row=r, column=0, columnspan=2)

    def _solvent_levels(self, col):
        for m in self.models.values():
            if col in m.get("category_levels", {}):
                return m["category_levels"][col]
        return []

    def _read_values(self):
        vals = {}
        for col, w in self.entries.items():
            s = w.get().strip()
            if s == "":
                messagebox.showwarning("提示", f"请填写 {col}")
                return None
            if isinstance(w, ttk.Combobox):
                vals[col] = s
            else:
                try:
                    vals[col] = float(s)
                except ValueError:
                    messagebox.showerror("错误", f"{col} 需为数值")
                    return None
        return vals

    def predict(self):
        if not self.models:
            messagebox.showerror("错误", f"没有可用模型，请检查 '{MODEL_DIR}' 目录。")
            return

        vals = self._read_values()
        if vals is None:
            return

        lines = []
        for target, meta in self.models.items():
            try:
                X = build_input_df(vals, meta)
                y = float(meta["model"].predict(X)[0])
                lines.append(f"{target}: {y:.3f}")
            except Exception as e:
                lines.append(f"{target}: 预测失败 ({e})")

        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("end", "\n".join(lines))
        self.out.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()