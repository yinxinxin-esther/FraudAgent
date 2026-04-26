# -*- coding: utf-8 -*-
"""
将用户反馈数据合并到训练集，重新训练 SVM 模型
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib

FEEDBACK_FILE = "user_feedback.csv"
ORIGINAL_TRAIN_FILE = "annotation_sample_v1.csv"  # 你原始的标注文件
COMBINED_TRAIN_FILE = "annotation_combined.csv"

# ---------- 文本清洗函数（与 app.py 保持一致）----------
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

def load_and_merge():
    """加载原始数据和反馈数据，合并清洗"""
    # 原始数据
    if os.path.exists(ORIGINAL_TRAIN_FILE):
        original = pd.read_csv(ORIGINAL_TRAIN_FILE, encoding="gb18030")
        # 确保有 clean_text，如果没有则生成
        if "clean_text" not in original.columns:
            if "raw_text" in original.columns:
                original["clean_text"] = original["raw_text"].apply(clean_text)
            else:
                raise ValueError("原始训练文件缺少文本列")
        original = original[['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags']].copy()
    else:
        original = pd.DataFrame(columns=['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags'])

    # 反馈数据
    if os.path.exists(FEEDBACK_FILE):
        feedback = pd.read_csv(FEEDBACK_FILE)
        # 反馈文件里已经有 clean_text，直接使用
        feedback = feedback[['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags']].copy()
    else:
        feedback = pd.DataFrame(columns=['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags'])

    combined = pd.concat([original, feedback], ignore_index=True)
    combined.to_csv(COMBINED_TRAIN_FILE, index=False, encoding="utf-8-sig")
    print(f"合并完成：原始 {len(original)} 条 + 反馈 {len(feedback)} 条 = 总计 {len(combined)} 条")
    return combined

def train_level1(df):
    """训练一级分类器"""
    df = df[df['text_role'].notna() & (df['text_role'] != "")]
    X = df['clean_text'].fillna("")
    y = df['text_role'].astype(str)

    base = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])
    model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    model.fit(X, y)
    return model

def train_level2(df):
    """训练二级分类器"""
    df = df[(df['text_role'] == 'direct_scam') & df['primary_class'].notna() & (df['primary_class'] != "")]
    if len(df) < 50 or df['primary_class'].nunique() < 2:
        print("二级分类器样本不足，跳过训练。")
        return None
    X = df['clean_text'].fillna("")
    y = df['primary_class'].astype(str)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])
    model.fit(X, y)
    return model

if __name__ == "__main__":
    print("正在合并数据...")
    combined_df = load_and_merge()

    print("正在训练一级模型...")
    level1 = train_level1(combined_df)
    joblib.dump(level1, "level1_model.pkl")
    print("一级模型已保存为 level1_model.pkl")

    print("正在训练二级模型...")
    level2 = train_level2(combined_df)
    if level2:
        joblib.dump(level2, "level2_model.pkl")
        print("二级模型已保存为 level2_model.pkl")

    print("\n✅ 重训练完成！请重启 Streamlit 应用以加载新模型。")