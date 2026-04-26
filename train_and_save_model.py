# -*- coding: utf-8 -*-
"""
训练 SVM 模型：text_role、primary_class、tags
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import joblib

SEED = 42
np.random.seed(SEED)

LABELED_PATH = "annotation_sample_v1.csv"   # 你的标注样本文件

# ---------- 文本清洗 ----------
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

# ---------- 读取数据 ----------
print("正在读取标注数据...")
df = pd.read_csv(LABELED_PATH, encoding="gb18030", encoding_errors="replace")

if "clean_text" not in df.columns:
    if "raw_text" in df.columns:
        df["clean_text"] = df["raw_text"].apply(clean_text)
    else:
        raise ValueError("标注文件中没有 clean_text 或 raw_text 列！")

# 统一 text_role 标签
role_map = {
    "direct_scam_text": "direct_scam",
    "direct_scam": "direct_scam",
    "scam_discussion": "scam_related_discussion",
    "scam_related_discussion": "scam_related_discussion",
    "non_scam": "non_scam",
    "insufficient_context": "context_insufficient",
    "context_insufficient": "context_insufficient",
}
df["text_role"] = df["text_role"].replace(role_map)

# ---------- 1. 训练 text_role 模型 ----------
print("训练 text_role 模型...")
level1_df = df[df["text_role"].notna() & (df["text_role"] != "")]
X1 = level1_df["clean_text"].fillna("")
y1 = level1_df["text_role"].astype(str)

base_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)),
    ("clf", LinearSVC(class_weight="balanced", random_state=SEED))
])
level1_model = CalibratedClassifierCV(base_pipeline, cv=3, method='sigmoid')
level1_model.fit(X1, y1)
joblib.dump(level1_model, "level1_model.pkl")
print("✅ level1_model.pkl 已保存")

# ---------- 2. 训练 primary_class 模型 ----------
print("训练 primary_class 模型...")
level2_df = df[(df["text_role"] == "direct_scam") & df["primary_class"].notna() & (df["primary_class"] != "")]
if len(level2_df) >= 50 and level2_df["primary_class"].nunique() >= 2:
    X2 = level2_df["clean_text"].fillna("")
    y2 = level2_df["primary_class"].astype(str)
    level2_model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
        ("clf", LinearSVC(class_weight="balanced", random_state=SEED))
    ])
    level2_model.fit(X2, y2)
    joblib.dump(level2_model, "level2_model.pkl")
    print("✅ level2_model.pkl 已保存")
else:
    print("⚠️ primary_class 样本不足，跳过训练。")

# ---------- 3. 训练 tags 多标签模型（新增）----------
print("训练 tags 多标签模型...")
tags_df = df[(df["text_role"] == "direct_scam") & df["tags"].notna() & (df["tags"] != "")]
if len(tags_df) >= 30:
    tags_df["tags_list"] = tags_df["tags"].apply(lambda x: [t.strip() for t in str(x).split(";") if t.strip()])
    mlb = MultiLabelBinarizer()
    y_tags = mlb.fit_transform(tags_df["tags_list"])
    print(f"标签类别数：{len(mlb.classes_)}")

    X_tags = tags_df["clean_text"].fillna("")
    tags_model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
        ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced", random_state=SEED)))
    ])
    tags_model.fit(X_tags, y_tags)

    joblib.dump(tags_model, "tags_model.pkl")
    joblib.dump(mlb, "tags_mlb.pkl")
    print("✅ tags_model.pkl 和 tags_mlb.pkl 已保存")
else:
    print("⚠️ tags 样本不足（<30条），跳过训练。")

print("\n训练完成！")