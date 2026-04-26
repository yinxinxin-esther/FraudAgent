# ==================== 诈骗话术策略分析====================

# -------------------- 0. 全局设置 --------------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 字体管理神器
import seaborn as sns
import pandas as pd
import random

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
sns.set_style("whitegrid")
random.seed(42)


# ---------- 自动匹配中文字体 ----------
def auto_set_chinese_font():
    candidate_fonts = [
        'SimHei',  # Windows 黑体（最常见）
        'Microsoft YaHei',  # Windows 微软雅黑
        'PingFang SC',  # Mac 苹果系统
        'Arial Unicode MS',  # Mac 备用
        'WenQuanYi Zen Hei',  # Linux
        'Noto Sans CJK SC',  # Linux 通用中文字体
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    selected = None
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            selected = font_name
            break

    if selected:
        plt.rcParams['font.sans-serif'] = [selected]
        print(f"已自动选择中文字体：{selected}")
    else:
        print("警告：未找到任何已知中文字体，图中文字可能仍为方框。")
        print("请搜索 “你的系统 安装中文字体” 来添加字体。")


auto_set_chinese_font()

# -------------------- 1. 数据读入与快速检查 --------------------
print("正在读取数据...")
try:
    df = pd.read_csv('fraud_strategy_final_coded_segments.csv')
    print(f"读取成功，数据维度：{df.shape[0]} 行，{df.shape[1]} 列")
except FileNotFoundError:
    print("错误：请确保 fraud_strategy_final_coded_segments.csv 在相同目录下！")
    exit()

# 查看必要的列是否存在
required_cols = ['final_text_role', 'final_scam_type', 'final_script_stage',
                 'final_tactic_tags', 'final_operation_tags', 'segment_text']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"警告：缺少关键列 {missing}，请检查文件名或列名")
else:
    print("关键列检查通过。")

# -------------------- 2. 数据清洗：多标签列拆分为列表 --------------------
print("正在清洗多标签列...")

# 将分号分隔的字符串转换为列表，方便统计
def split_tags(tag_str):
    if pd.isna(tag_str) or str(tag_str).strip() == '':
        return []
    return [t.strip() for t in str(tag_str).split(';')]

df['tactic_list'] = df['final_tactic_tags'].apply(split_tags)
df['operation_list'] = df['final_operation_tags'].apply(split_tags)

# -------------------- 3. 战术标签频次分析 --------------------
print("\n========== 1. 战术标签频次排名 ==========")

# 展开列表，每个标签一行
tactic_exploded = df.explode('tactic_list')
tactic_counts = tactic_exploded['tactic_list'].value_counts()
print(tactic_counts.head(10).to_string())

# 画图：Top10 战术
plt.figure(figsize=(10, 6))
top10_tactics = tactic_counts.head(10)
sns.barplot(x=top10_tactics.values, y=top10_tactics.index, palette='viridis')
plt.title('最常见的 10 种诈骗战术')
plt.xlabel('出现次数')
plt.ylabel('战术标签')
plt.tight_layout()
plt.savefig('tactic_top10.png', dpi=150)
plt.show()

# -------------------- 4. 操作标签频次分析 --------------------
print("\n========== 2. 操作标签频次排名 ==========")

operation_exploded = df.explode('operation_list')
operation_counts = operation_exploded['operation_list'].value_counts()
print(operation_counts.head(10).to_string())

plt.figure(figsize=(10, 6))
top10_ops = operation_counts.head(10)
sns.barplot(x=top10_ops.values, y=top10_ops.index, palette='coolwarm')
plt.title('最常见的 10 种操作动作')
plt.xlabel('频次')
plt.ylabel('操作标签')
plt.tight_layout()
plt.savefig('operation_top10.png', dpi=150)
plt.show()

# -------------------- 5. 剧本阶段分布 --------------------
print("\n========== 3. 剧本阶段分布 ==========")

stage_counts = df['final_script_stage'].value_counts()
print(stage_counts.to_string())

# 饼图（只展示主要阶段，其余合并为“其他”）
top_stages = stage_counts.head(6).copy()
other_sum = stage_counts.iloc[6:].sum() if len(stage_counts) > 6 else 0
if other_sum > 0:
    top_stages['其他'] = other_sum

plt.figure(figsize=(8, 8))
plt.pie(top_stages.values, labels=top_stages.index, autopct='%1.1f%%',
        startangle=140, textprops={'fontsize': 11})
plt.title('诈骗剧本阶段分布')
plt.tight_layout()
plt.savefig('stage_pie.png', dpi=150)
plt.show()

# -------------------- 6. 诈骗类型 vs 战术交叉分析 --------------------
print("\n========== 4. 不同诈骗类型的战术偏好 ==========")

# 取每行第一个战术作为主战术（简化，也可全展开）
df['primary_tactic'] = df['tactic_list'].apply(lambda x: x[0] if len(x) > 0 else 'other')

# 生成交叉表
cross_tab = pd.crosstab(df['final_scam_type'], df['primary_tactic'])
# 计算行百分比（每个诈骗类型内，各战术占比）
cross_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

# 画热力图
plt.figure(figsize=(14, 10))
sns.heatmap(cross_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, linecolor='gray')
plt.title('各诈骗类型的战术偏好（行百分比）')
plt.xlabel('主战术标签')
plt.ylabel('诈骗类型')
plt.tight_layout()
plt.savefig('scamtype_tactic_heatmap.png', dpi=150)
plt.show()

# -------------------- 7. 典型话术采样 --------------------
print("\n========== 5. 典型话术示例（前10战术） ==========")

samples = []
for tactic in tactic_counts.head(10).index:
    subset = df[df['final_tactic_tags'].str.contains(tactic, na=False)]
    if len(subset) == 0:
        continue
    if len(subset) >= 3:
        texts = subset['segment_text'].sample(3, random_state=42).tolist()
    else:
        texts = subset['segment_text'].tolist()
    samples.append({'战术': tactic, '示例话术': texts})

samples_df = pd.DataFrame(samples)
for i, row in samples_df.iterrows():
    print(f"\n【{row['战术']}】")
    for j, text in enumerate(row['示例话术'], 1):
        print(f"  {j}. {text}")
