# -*- coding: utf-8 -*-
"""
LLM Relabeler for low-confidence samples
"""

import json
import random
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# ==================== 配置区 ====================
API_KEY = "sk-3543d069c8f5463687ea83cedaddede3"
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

TRAIN_EXCEL_PATH = "800训练数据.xlsx"  # 用于 Few-shot 示例
LOW_CONF_FILE = "low_confidence_samples.csv"  # 上一步生成的文件
OUTPUT_FILE = "llm_relabeled.csv"  # 输出结果


# ===============================================

# ---------- 1. 从训练数据中抽取 Few-shot 示例 ----------
def load_few_shot_examples(excel_path, n=10):
    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    df = df[['clean_text', 'text_role', 'is_scam', 'primary_class', 'tags']]
    df = df.dropna(subset=['clean_text'])

    examples = []
    for _, row in df.sample(n=min(n, len(df)), random_state=42).iterrows():
        label = {
            "text_role": row['text_role'] if pd.notna(row['text_role']) else "",
            "is_scam": int(row['is_scam']) if pd.notna(row['is_scam']) else 0,
            "primary_class": row['primary_class'] if pd.notna(row['primary_class']) else "",
            "tags": row['tags'] if pd.notna(row['tags']) else ""
        }
        examples.append({"text": row['clean_text'], "label": label})
    return examples


# ---------- 2. 构建 Prompt ----------
SYSTEM_PROMPT = """你是一个专业的反诈骗内容分析专家。请根据用户提供的文本内容，完成以下标注任务：

1. text_role：判断文本属于哪一类：
   - direct_scam：直接实施诈骗的内容
   - scam_related_discussion：与诈骗相关的讨论、曝光、求助、新闻报道
   - non_scam：与诈骗无关的正常内容
   - context_insufficient：信息不足，无法判断

2. is_scam：若 text_role 为 direct_scam 则填 1，否则填 0。

3. primary_class：若为诈骗内容，归类到以下具体类型之一（非诈骗留空）：
   investment, relationship_trust, transaction_refund, credential_phishing, task_job, loan_account, extortion, prize_benefit

4. tags：提取关键诈骗手法或特征词，用分号(;)分隔。

请严格按照JSON格式输出，不要包含任何额外解释：
{"text_role": "", "is_scam": 0, "primary_class": "", "tags": ""}"""


def build_user_prompt(text, examples):
    example_str = ""
    for i, ex in enumerate(examples, 1):
        example_str += f"\n示例{i}:\n文本：{ex['text']}\n标注结果：{json.dumps(ex['label'], ensure_ascii=False)}\n"
    prompt = f"""以下是一些已标注的示例，请参考它们对最后一条文本进行标注。

{example_str}
现在请对以下文本进行标注：
文本：{text}

输出JSON："""
    return prompt


# ---------- 3. 调用 LLM ----------
def label_one_text(client, text, examples):
    if pd.isna(text) or str(text).strip() == "":
        return {"text_role": "context_insufficient", "is_scam": 0, "primary_class": "", "tags": ""}

    user_prompt = build_user_prompt(str(text), examples)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
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
        error_str = str(e)
        # 判断是否为内容审核失败 (data_inspection_failed)
        if "data_inspection_failed" in error_str or "inappropriate content" in error_str.lower():
            print(f"  [警告] 内容审核未通过，标记为 context_insufficient")
        else:
            print(f"标注出错: {e}")
        # 返回默认值，并标记为审核失败（可以加一个额外字段，但这里只返回基本标签）
        return {"text_role": "context_insufficient", "is_scam": 0, "primary_class": "", "tags": ""}


# ---------- 4. 主程序 ----------
def main():
    print("正在加载 Few-shot 示例...")
    examples = load_few_shot_examples(TRAIN_EXCEL_PATH, n=10)
    print(f"已加载 {len(examples)} 条示例。")

    print("正在初始化 LLM 客户端...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"正在读取低置信度样本文件: {LOW_CONF_FILE}")
    low_conf_df = pd.read_csv(LOW_CONF_FILE)
    print(f"共有 {len(low_conf_df)} 条样本待重新标注。")

    results = []
    for idx, row in tqdm(low_conf_df.iterrows(), total=len(low_conf_df), desc="LLM 重新标注中"):
        text = row['clean_text']
        label = label_one_text(client, text, examples)

        new_row = row.to_dict()
        new_row['llm_text_role'] = label['text_role']
        new_row['llm_is_scam'] = label['is_scam']
        new_row['llm_primary_class'] = label['primary_class']
        new_row['llm_tags'] = label['tags']
        results.append(new_row)

        time.sleep(0.5)  # 避免请求过快

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n标注完成！结果已保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()