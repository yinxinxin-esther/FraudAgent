# -*- coding: utf-8 -*-
"""
诈骗文本智能标注 Agent - 网页版（含 tags 预测与自动重训练）
"""

import json
import re
import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# ==================== 配置区 ====================
API_KEY = "sk-3543d069c8f5463687ea83cedaddede3"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

CONFIDENCE_THRESHOLD = 0.55
FEEDBACK_FILE = "user_feedback.csv"
ORIGINAL_TRAIN_FILE = "annotation_sample_v1.csv"
COMBINED_TRAIN_FILE = "annotation_combined.csv"
RETRAIN_THRESHOLD = 10
# ================================================

st.set_page_config(page_title="诈骗文本智能标注 Agent", page_icon="🛡️", layout="wide")

# ---------- 文本清洗 ----------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
HTML_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\S+")
PHONE_PATTERN = re.compile(r"\b\d{6,}\b")

def clean_text(text):
    if not text: return ""
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
@st.cache_resource
def load_models():
    models = {}
    models["level1"] = joblib.load("level1_model.pkl")
    try:
        models["level2"] = joblib.load("level2_model.pkl")
    except:
        models["level2"] = None
    try:
        models["tags_model"] = joblib.load("tags_model.pkl")
        models["mlb"] = joblib.load("tags_mlb.pkl")
    except:
        models["tags_model"] = None
        models["mlb"] = None
    return models

models = load_models()
level1_model = models["level1"]
level2_model = models["level2"]
tags_model = models["tags_model"]
mlb = models["mlb"]

def predict_tags(text):
    if tags_model is None or mlb is None:
        return ""
    cleaned = clean_text(text)
    pred = tags_model.predict([cleaned])[0]
    tags_list = mlb.inverse_transform(np.array([pred]))[0]
    return ";".join(tags_list)

# ---------- 训练函数（用于重训练）----------
def train_level1(df):
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
    df = df[(df['text_role'] == 'direct_scam') & df['primary_class'].notna() & (df['primary_class'] != "")]
    if len(df) < 50 or df['primary_class'].nunique() < 2:
        return None
    X = df['clean_text'].fillna("")
    y = df['primary_class'].astype(str)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
        ("clf", LinearSVC(class_weight="balanced", random_state=42))
    ])
    model.fit(X, y)
    return model

def train_tags_model(df):
    df = df[(df['text_role'] == 'direct_scam') & df['tags'].notna() & (df['tags'] != "")]
    if len(df) < 30:
        return None, None
    df["tags_list"] = df["tags"].apply(lambda x: [t.strip() for t in str(x).split(";") if t.strip()])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["tags_list"])
    X = df['clean_text'].fillna("")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)),
        ("clf", OneVsRestClassifier(LinearSVC(class_weight="balanced", random_state=42)))
    ])
    model.fit(X, y)
    return model, mlb

def perform_retraining():
    try:
        if os.path.exists(ORIGINAL_TRAIN_FILE):
            original = pd.read_csv(ORIGINAL_TRAIN_FILE, encoding="gb18030")
            if "clean_text" not in original.columns:
                if "raw_text" in original.columns:
                    original["clean_text"] = original["raw_text"].apply(clean_text)
                else:
                    st.error("原始训练文件缺少文本列")
                    return False
            original = original[['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags']].copy()
        else:
            original = pd.DataFrame(columns=['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags'])

        if os.path.exists(FEEDBACK_FILE):
            feedback = pd.read_csv(FEEDBACK_FILE)
            feedback = feedback[['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags']].copy()
        else:
            feedback = pd.DataFrame(columns=['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags'])

        combined = pd.concat([original, feedback], ignore_index=True)
        combined.to_csv(COMBINED_TRAIN_FILE, index=False, encoding="utf-8-sig")

        with st.spinner("训练一级模型..."):
            level1 = train_level1(combined)
            joblib.dump(level1, "level1_model.pkl")
        st.success("✅ 一级模型已更新")

        with st.spinner("训练二级模型..."):
            level2 = train_level2(combined)
            if level2:
                joblib.dump(level2, "level2_model.pkl")
                st.success("✅ 二级模型已更新")
            else:
                st.warning("⚠️ 二级模型样本不足，未更新")

        with st.spinner("训练 tags 模型..."):
            tags_mdl, tags_mlb = train_tags_model(combined)
            if tags_mdl:
                joblib.dump(tags_mdl, "tags_model.pkl")
                joblib.dump(tags_mlb, "tags_mlb.pkl")
                st.success("✅ tags 模型已更新")
            else:
                st.warning("⚠️ tags 样本不足，未更新")

        st.cache_resource.clear()
        return True
    except Exception as e:
        st.error(f"❌ 重训练失败：{e}")
        return False

# ---------- LLM 精判 ----------
SYSTEM_PROMPT = """你是一个专业的反诈骗内容分析专家。请根据用户提供的文本内容，完成以下标注任务：

1. text_role：direct_scam / scam_related_discussion / non_scam / context_insufficient
2. is_scam：若为 direct_scam 则填 1，否则 0。
3. primary_class：若为诈骗，可选：investment, relationship_trust, transaction_refund, credential_phishing, task_job, loan_account, extortion, prize_benefit，否则留空。
    建议先用这 8 类：
        investment：投资理财型 
        task_job：兼职刷单型 
        transaction_refund：交易退款型 
        loan_account：借贷解冻型 
        prize_benefit：领奖补贴型 
        relationship_trust：关系诱导型 
        credential_phishing：凭证窃取型 
        extortion：敲诈勒索型 
    怎么判只看一句话：
    骗子最终要你做什么？
        例如：
        恋爱聊天，最后让你买币赚钱 → investment 
        冒充客服，说订单异常让你退款操作 → transaction_refund 
        冒充银行，说账户异常让你点链接输验证码 → credential_phishing 
        说兼职做任务、垫资返佣 → task_job 
4. tags：tags 不是主类，可以多选。
    你先用最简单的自然语言标签就行，用分号隔开。
        例如：
        冒充银行;短信;验证码;紧迫感 
        冒充客服;退款;下载app 
        恋爱关系;投资;高收益 
        兼职;垫付;返佣 
    建议标签先围绕四类写：
        1.冒充对象：银行 / 客服 / 政府 / 亲友 / 平台 
        2.渠道：短信 / 电话 / 社媒 / 邮件 / 假网站 
        3.动作：转账 / 验证码 / 链接 / 下载App / 充值 
        4.心理策略：紧迫感 / 权威 / 恐惧 / 信任 / 利诱 

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
        st.warning(f"LLM调用失败：{e}")
        return {"text_role": "context_insufficient", "is_scam": 0, "primary_class": "", "tags": ""}

def predict(text):
    cleaned = clean_text(text)
    if not cleaned:
        return {"text_role": "context_insufficient", "is_scam": 0, "primary_class": "", "tags": "", "confidence": 0.0, "source": "N/A"}

    probs = level1_model.predict_proba([cleaned])[0]
    confidence = np.max(probs)
    svm_role = level1_model.predict([cleaned])[0]

    result = {
        "text_role": svm_role,
        "is_scam": 1 if svm_role == "direct_scam" else 0,
        "primary_class": "",
        "tags": "",
        "confidence": confidence,
        "source": "SVM"
    }

    if svm_role == "direct_scam":
        if level2_model is not None:
            result['primary_class'] = level2_model.predict([cleaned])[0]
        result['tags'] = predict_tags(text)

    if confidence < CONFIDENCE_THRESHOLD:
        llm_result = llm_relabel(text)
        result['text_role'] = llm_result['text_role']
        result['is_scam'] = llm_result['is_scam']
        if llm_result.get('primary_class'):
            result['primary_class'] = llm_result['primary_class']
        # 注意：tags 可以保留本地模型结果，不用LLM覆盖会保证稳定
        result['tags'] = llm_result.get('tags', result['tags'])
        result['source'] = 'LLM'

    return result

# ---------- 保存反馈 ----------
def save_feedback(original_text, predicted_result, corrected_result):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_text": original_text,
        "clean_text": clean_text(original_text),
        "text_role": corrected_result["text_role"],
        "is_scam": corrected_result["is_scam"],
        "primary_class": corrected_result["primary_class"],
        "tags": corrected_result["tags"],
        "predicted_text_role": predicted_result["text_role"],
        "predicted_is_scam": predicted_result["is_scam"],
        "predicted_primary_class": predicted_result["primary_class"],
        "predicted_tags": predicted_result["tags"],
        "source": predicted_result.get("source", ""),
        "confidence": predicted_result.get("confidence", 0.0)
    }
    df = pd.DataFrame([record])
    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")

# ---------- 中文映射 ----------
ROLE_MAP_CN = {
    "direct_scam": "直接诈骗",
    "scam_related_discussion": "诈骗讨论",
    "non_scam": "非诈骗",
    "context_insufficient": "信息不足"
}
PRIMARY_CLASS_MAP_CN = {
    "investment": "投资诈骗",
    "relationship_trust": "情感信任诈骗",
    "transaction_refund": "交易退款诈骗",
    "credential_phishing": "钓鱼盗号",
    "task_job": "任务/求职诈骗",
    "loan_account": "贷款/账户诈骗",
    "extortion": "敲诈勒索",
    "prize_benefit": "中奖/福利诈骗",
    "": "其他"
}
ROLE_MAP_EN = {v: k for k, v in ROLE_MAP_CN.items()}
PRIMARY_CLASS_MAP_EN = {v: k for k, v in PRIMARY_CLASS_MAP_CN.items()}
PRIMARY_CLASS_MAP_EN["其他"] = ""

# ---------- 初始化 session_state ----------
if "input_text_value" not in st.session_state:
    st.session_state.input_text_value = ""          # 用于绑定 text_area 的 value
if "result" not in st.session_state:
    st.session_state.result = None
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

# ---------- 界面 ----------
st.title("🛡️ 诈骗文本智能标注 Agent")
st.markdown("输入任意文本，系统将自动判断其是否涉及诈骗，并输出详细标签。")

with st.sidebar:
    st.header("ℹ️ 标签说明")
    st.subheader("📌 primary_class（诈骗细类）")
    st.markdown("""
    - **投资诈骗**：虚假投资平台、杀猪盘、高回报理财等
    - **情感信任诈骗**：网恋交友、冒充亲友、建立信任后骗钱
    - **交易退款诈骗**：虚假购物、退款陷阱、货到付款诈骗
    - **钓鱼盗号**：仿冒网站、虚假客服、诱导输入账号密码
    - **任务/求职诈骗**：刷单兼职、虚假招聘、垫付资金任务
    - **贷款/账户诈骗**：虚假贷款、账户异常、要求缴纳保证金
    - **敲诈勒索**：裸聊敲诈、隐私威胁、恐吓索财
    - **中奖/福利诈骗**：虚假中奖信息、免费赠品、诱导付运费
    - **其他**：不属于以上任何一类
    """)
    st.subheader("🏷️ tags（关键标签）")
    st.markdown("提取文本中的核心诈骗手法或特征词，如：杀猪盘、冒充客服、钓鱼链接等。")
    st.divider()
    st.metric("置信度阈值", f"{CONFIDENCE_THRESHOLD*100:.0f}%")
    st.divider()
    st.markdown("### 📊 模型状态")
    st.success("✅ 一级模型已加载")
    if level2_model: st.success("✅ 二级模型已加载")
    else: st.warning("⚠️ 二级模型未加载")
    st.success("✅ tags 模型已加载")
    st.divider()
    st.markdown("### 🔧 模型重训练")
    fb_count = len(pd.read_csv(FEEDBACK_FILE)) if os.path.exists(FEEDBACK_FILE) else 0
    col1, col2 = st.columns(2)
    col1.metric("📝 反馈数", fb_count)
    col2.metric("🎯 阈值", RETRAIN_THRESHOLD)
    if fb_count >= RETRAIN_THRESHOLD:
        st.success("✅ 可重训练")
        disabled = False
    else:
        st.info(f"⏳ 还需 {RETRAIN_THRESHOLD - fb_count} 条")
        disabled = True
    if st.button("🔄 立即重训练并更新模型", disabled=disabled, use_container_width=True):
        with st.status("重训练中...", expanded=True) as status:
            if perform_retraining():
                status.update(label="重训练完成！", state="complete")
                st.success("模型已更新，即将刷新...")
                time.sleep(2)
                st.rerun()
            else:
                status.update(label="重训练失败", state="error")

# 输入框，绑定 session_state.input_text_value
user_input = st.text_area(
    "📝 请输入待分析的文本：",
    height=200,
    value=st.session_state.input_text_value,
    key="text_area_widget"  # 注意：key不能与value的变量同名
)

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    if st.button("🔍 开始分析", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("分析中..."):
                result = predict(user_input.strip())
                st.session_state.result = result
                st.session_state.original_text = user_input.strip()
                st.session_state.show_feedback = False
        else:
            st.warning("请输入文本")
with col2:
    if st.button("🔄 重置", use_container_width=True):
        st.session_state.input_text_value = ""   # 清空绑定的值
        st.session_state.result = None           # 清空结果
        st.session_state.show_feedback = False
        st.rerun()

# 显示结果
if st.session_state.result:
    result = st.session_state.result
    original_text = st.session_state.original_text
    st.divider()
    st.subheader("📋 分析结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("文本类型", ROLE_MAP_CN.get(result['text_role'], result['text_role']))
    c2.metric("是否诈骗术语", "是" if result['is_scam'] else "否")
    c3.metric("诈骗细类", PRIMARY_CLASS_MAP_CN.get(result.get('primary_class',''), result.get('primary_class','') or "其他"))
    c4.metric("判断来源", "大模型" if result.get('source')=='LLM' else "SVM模型")
    st.markdown("**🏷️ 关键标签：**")
    tags_val = result.get('tags','')
    if tags_val:
        tags_html = " ".join([f'<span style="background:#f0f2f6;padding:4px 12px;border-radius:16px;margin-right:8px;">{t.strip()}</span>' for t in tags_val.split(';') if t.strip()])
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.caption("无")
    conf = result.get('confidence', 0.0)
    st.markdown(f"**📊 置信度：** {conf:.2%}")
    st.progress(conf)
    with st.expander("🔎 JSON"):
        st.json(result)
    st.divider()
    if st.button("✏️ 不满意？手动修正"):
        st.session_state.show_feedback = not st.session_state.show_feedback
        st.rerun()

    if st.session_state.show_feedback:
        with st.form("feedback"):
            cur_role = ROLE_MAP_CN.get(result['text_role'], result['text_role'])
            cur_is_scam = result['is_scam']
            cur_primary = PRIMARY_CLASS_MAP_CN.get(result.get('primary_class',''), result.get('primary_class','') or "其他")
            cur_tags = result.get('tags','')
            ca, cb = st.columns(2)
            with ca:
                new_role = st.selectbox("文本类型", list(ROLE_MAP_CN.values()), index=list(ROLE_MAP_CN.values()).index(cur_role) if cur_role in ROLE_MAP_CN.values() else 0)
                new_is_scam = st.radio("是否诈骗", [0,1], format_func=lambda x:"是" if x else "否", index=cur_is_scam)
            with cb:
                new_primary = st.selectbox("诈骗细类", list(PRIMARY_CLASS_MAP_CN.values()), index=list(PRIMARY_CLASS_MAP_CN.values()).index(cur_primary) if cur_primary in PRIMARY_CLASS_MAP_CN.values() else 0)
                new_tags = st.text_input("关键标签（分号分隔）", value=cur_tags)
            if st.form_submit_button("📤 提交反馈"):
                corrected = {
                    "text_role": ROLE_MAP_EN[new_role],
                    "is_scam": new_is_scam,
                    "primary_class": PRIMARY_CLASS_MAP_EN.get(new_primary, ""),
                    "tags": new_tags
                }
                save_feedback(original_text, result, corrected)
                st.success("✅ 反馈已保存")
                st.session_state.show_feedback = False
                time.sleep(0.5)
                st.rerun()

st.divider()
st.caption("💡 提示：SVM快速判断，低置信度时启用大模型精判")
