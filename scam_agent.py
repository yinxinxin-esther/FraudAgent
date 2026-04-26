# -*- coding: utf-8 -*-
"""
诈骗文本标注 Agent
"""

import json
import re
import joblib
import numpy as np
from openai import OpenAI

# ==================== 配置区 ====================
API_KEY = "sk-3543d069c8f5463687ea83cedaddede3"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

CONFIDENCE_THRESHOLD = 0.6
# ===============================================

# ---------- 加载模型 ----------
print("加载模型中...")
level1_model = joblib.load("level1_model.pkl")
try:
    level2_model = joblib.load("level2_model.pkl")
except:
    level2_model = None
    print("未找到二级模型，将只输出一级分类。")

# ---------- 文本清洗函数 ----------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
HTML_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\S+")
PHONE_PATTERN = re.compile(r"\b\d{6,}\b")


def clean_text(text):
    if not text:
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


# ---------- LLM 精判函数 ----------
SYSTEM_PROMPT = """你是一个专业的反诈骗内容分析专家。请根据用户提供的文本内容，完成以下标注任务：

1. text_role：direct_scam / scam_related_discussion / non_scam / context_insufficient
2. is_scam：若为 direct_scam 则填 1，否则 0。
3. primary_class：若为诈骗，可选：investment, relationship_trust, transaction_refund, credential_phishing, task_job, loan_account, extortion, prize_benefit，否则留空。
4. tags：关键手法或特征词，用分号分隔。

输出JSON格式：{"text_role": "", "is_scam": 0, "primary_class": "", "tags": ""}"""


def llm_relabel(text):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"文本：{text}\n输出JSON："}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result.setdefault('text_role', 'context_insufficient')
        result.setdefault('is_scam', 0)
        result.setdefault('primary_class', '')
        result.setdefault('tags', '')
        return result
    except Exception as e:
        print(f"LLM调用失败: {e}")
        return {"text_role": "context_insufficient", "is_scam": 0, "primary_class": "", "tags": ""}


# ---------- 主预测函数 ----------
def predict(text):
    cleaned = clean_text(text)
    if not cleaned:
        return {"text_role": "context_insufficient", "is_scam": 0, "primary_class": "", "tags": ""}

    # SVM 预测置信度
    probs = level1_model.predict_proba([cleaned])[0]
    confidence = np.max(probs)
    svm_role = level1_model.predict([cleaned])[0]

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"   [SVM置信度={confidence:.2f}，低于阈值，启用LLM精判...]")
        llm_result = llm_relabel(text)
        if llm_result['text_role'] == 'direct_scam' and level2_model is not None:
            llm_result['primary_class'] = level2_model.predict([cleaned])[0]
        return llm_result
    else:
        result = {
            "text_role": svm_role,
            "is_scam": 1 if svm_role == "direct_scam" else 0,
            "primary_class": "",
            "tags": ""
        }
        if svm_role == "direct_scam" and level2_model is not None:
            result['primary_class'] = level2_model.predict([cleaned])[0]
        return result


# ---------- 交互式命令行 ----------
if __name__ == "__main__":
    print("=" * 50)
    print("诈骗文本标注 Agent 已启动")
    print("输入文本后按回车，输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n请输入文本: ").strip()
        if user_input.lower() == 'quit':
            print("再见！")
            break
        if not user_input:
            continue

        result = predict(user_input)
        print("\n--- 标注结果 ---")
        print(f"text_role: {result['text_role']}")
        print(f"is_scam: {result['is_scam']}")
        print(f"primary_class: {result['primary_class']}")
        print(f"tags: {result['tags']}")