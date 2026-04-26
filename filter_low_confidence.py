# -*- coding: utf-8 -*-
"""
对已打标的 posts_with_predictions.csv 计算置信度，筛选低置信度样本。
"""

import re
import numpy as np
import pandas as pd
import joblib

# ---------- 文本清洗函数（保持一致）----------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
HTML_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\S+")
PHONE_PATTERN = re.compile(r"\b\d{6,}\b")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    text = re.sub(r"[-_/=\\]+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- 加载模型 ----------
print("加载模型中...")
level1_model = joblib.load("level1_model.pkl")

# ---------- 读取已打标数据 ----------
print("读取 posts_with_predictions.csv...")
df = pd.read_csv("posts_with_predictions.csv")

# 确保有 clean_text 列，如果没有，尝试从 raw_text 生成
if "clean_text" not in df.columns:
    if "raw_text" in df.columns:
        print("未找到 clean_text 列，正在从 raw_text 生成...")
        df["clean_text"] = df["raw_text"].apply(clean_text)
    else:
        raise ValueError("数据中既没有 clean_text 也没有 raw_text，无法继续。")

# ---------- 计算置信度 ----------
print("正在计算每条文本的置信度...")
texts = df["clean_text"].fillna("")
probs = level1_model.predict_proba(texts)
max_probs = np.max(probs, axis=1)
df["confidence"] = max_probs

# ---------- 筛选低置信度样本 ----------
CONFIDENCE_THRESHOLD = 0.6   # 你可以调整这个值
low_conf_mask = df["confidence"] < CONFIDENCE_THRESHOLD
low_conf_df = df[low_conf_mask].copy()

print(f"共 {len(df)} 条数据，其中低置信度样本 {len(low_conf_df)} 条 ({len(low_conf_df)/len(df)*100:.1f}%)")

# 保存低置信度样本
low_conf_df.to_csv("low_confidence_samples.csv", index=False, encoding="utf-8-sig")
print("低置信度样本已保存为 low_confidence_samples.csv")

# 同时把带置信度的完整数据也保存一份（可选）
df.to_csv("posts_with_confidence.csv", index=False, encoding="utf-8-sig")
print("带置信度的完整数据已保存为 posts_with_confidence.csv")