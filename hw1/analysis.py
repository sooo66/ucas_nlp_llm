import json
import random
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
import nltk
import jieba
import matplotlib.font_manager as fm
from matplotlib import rcParams
from scipy.stats import skew  # 新增导入，用于分布分析
import zhplot

# 下载必要的资源
nltk.download('punkt', quiet=True)

# 字体设置
font_candidates = ['Noto Sans CJK TC', 'Noto Sans CJK SC', 'DejaVu Sans', 'Arial Unicode', 'SimHei']
for font in font_candidates:
    if font in [f.name for f in fm.fontManager.ttflist]:
        rcParams['font.sans-serif'] = [font]
        rcParams['axes.unicode_minus'] = False
        break

# 数据路径
EN_CORPUS_PATH = "data/clean/en_wiki/en_wikipedia_clean.jsonl"
ZH_CORPUS_PATH = "data/clean/zh_wiki/zh_wikipedia_clean.jsonl"  # 修正为正确的中文路径

# 实验规模
SCALES = [10_000, 50_000, 1_000_000, 5_000_000, 16_000_000]

# 分析粒度配置
ANALYSIS_LEVELS = {
    "Char-EN": {"lang": "en", "level": "char", "vocab_filter": lambda c: c.islower() and c.isalpha()},
    "Word-EN": {"lang": "en", "level": "word", "tokenizer": nltk.word_tokenize},
    "Char-ZH": {"lang": "zh", "level": "char", "vocab_filter": lambda c: '\u4e00' <= c <= '\u9fff'},
    "Word-ZH": {"lang": "zh", "level": "word", "tokenizer": jieba.cut}
}

# 结果存储
results = []

def load_corpus(path):
    """加载 JSONL 语料，每行是一个 JSON 对象，至少包含 'text' 字段"""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'content' in data:
                    texts.append(data['content'])
            except json.JSONDecodeError:
                continue
    return texts

def sample_tokens(tokens, n):
    """从 token 列表中随机采样 n 个（允许重复采样以支持大 N）"""
    if len(tokens) >= n:
        return random.sample(tokens, n)
    else:
        return random.choices(tokens, k=n)

def tokenize_en_char(text):
    return [c for c in text.lower() if c.isalpha() and c.islower()]

def tokenize_en_word(text):
    return [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]

def tokenize_zh_char(text):
    return [c for c in text if '\u4e00' <= c <= '\u9fff']

def tokenize_zh_word(text):
    return list(jieba.cut(text))

def get_tokenizer(level_config, text):
    if level_config["level"] == "char" and level_config["lang"] == "en":
        return tokenize_en_char(text)
    elif level_config["level"] == "word" and level_config["lang"] == "en":
        return tokenize_en_word(text)
    elif level_config["level"] == "char" and level_config["lang"] == "zh":
        return tokenize_zh_char(text)
    elif level_config["level"] == "word" and level_config["lang"] == "zh":
        return tokenize_zh_word(text)
    return []

def compute_entropy(freq_counter, total):
    """计算信息熵 H(X) = -sum p(x) log2 p(x)"""
    entropy = 0.0
    for count in freq_counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def fit_zipf(freq_counter):
    """拟合 Zipf 定律：rank vs frequency，log-log 线性回归，只使用前100个高频项"""
    if len(freq_counter) < 2:
        return None, None, None, None
    # 按频率排序，取前100
    sorted_items = sorted(freq_counter.items(), key=lambda x: -x[1])[:100]
    if len(sorted_items) < 2:
        return None, None, None, None
    ranks = np.arange(1, len(sorted_items) + 1)
    freqs = np.array([count for _, count in sorted_items])
    
    # 取 log
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    
    # 线性回归
    slope, intercept, r_value, _, _ = linregress(log_ranks, log_freqs)
    return slope, intercept, r_value**2, list(zip([item[0] for item in sorted_items], freqs))

def plot_zipf_multi(rank_freq_dict, scale, output_dir="figures"):
    """
    rank_freq_dict: {level: [(token, freq), ...]}   # 已经排好序的前 20 项
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Zipf's Law (N = {scale:,})", fontsize=16, fontweight='bold')
    
    levels = ["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    titles = ["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    
    for ax, level, ctitle in zip(axes.flatten(), levels, titles):
        if level not in rank_freq_dict:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ctitle)
            continue
        
        ranks, freqs = zip(*[(i+1, f) for i, (_, f) in enumerate(rank_freq_dict[level])])
        ax.loglog(ranks, freqs, marker='o', linestyle='-', markersize=6, label=level.split("-")[0])
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(ctitle)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        # 标注斜率（如果已经在 results 中算好）
        slope = next((r["Zipf_Slope"] for r in results
                      if r["Scale"] == scale and r["Level"] == level), None)
        if slope:
            ax.text(0.05, 0.85, f"slope ≈ {slope:.3f}", transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/zipf_compare_N{scale}.png", dpi=300)
    plt.close()

def plot_zipf_across_scales(all_rank_freq, output_dir="figures"):
    """
    all_rank_freq: {level: {scale: [(token, freq), ...]}}
    绘制同一粒度在不同 scale 下的对比（2×2 子图）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Zipf's Law 随数据规模演化", fontsize=16, fontweight='bold')
    
    levels = ["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    titles =["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(SCALES)))
    
    for ax, level, ctitle in zip(axes.flatten(), levels, titles):
        for scale, color in zip(SCALES, colors):
            if scale not in all_rank_freq.get(level, {}):
                continue
            ranks, freqs = zip(*[(i+1, f) for i, (_, f) in enumerate(all_rank_freq[level][scale])])
            ax.loglog(ranks, freqs, marker='o', linestyle='-', color=color,
                      label=f"N={scale//1000 if scale<1_000_000 else scale//1_000_000}M")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(ctitle)
        ax.legend(title="Scale", loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/zipf_evolution.png", dpi=300)
    plt.close()

def compute_prob_stats(counter, total_n, top_n=10):
    """计算概率统计"""
    if total_n == 0:
        return {}, 0.0, 0.0, 0.0
    
    # 计算概率字典
    probs = {token: count / total_n for token, count in counter.items()}
    
    # 排序
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    
    # top1 概率
    top1_prob = sorted_probs[0][1] if sorted_probs else 0.0
    
    # 累计前top_n概率
    cum_top_n = sum(p for _, p in sorted_probs[:top_n])
    
    # 分布偏度（使用所有概率值）
    prob_values = np.array(list(probs.values()))
    prob_skew = skew(prob_values) if len(prob_values) > 2 else 0.0
    
    # 返回top20概率用于绘图（token, prob）
    top20_probs = sorted_probs[:20]  # 调整为20以便绘图
    
    return top20_probs, top1_prob, cum_top_n, prob_skew

def plot_probabilities_multi(prob_dict, scale, output_dir="figures"):
    """
    prob_dict: {level: [(token, prob), ...]}  # 已排序的前20项
    绘制top 20概率柱状图（2x2子图）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Top Token Probabilities (N = {scale:,})", fontsize=16, fontweight='bold')
    
    levels = ["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    titles = ["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    
    for ax, level, ctitle in zip(axes.flatten(), levels, titles):
        if level not in prob_dict:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(ctitle)
            continue
        
        tokens, probs = zip(*prob_dict[level][:20])  # 只画top20
        ax.bar(tokens, probs, color='skyblue')
        ax.set_xlabel("Token")
        ax.set_ylabel("Probability")
        ax.set_title(ctitle)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', ls="--", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/prob_top_N{scale}.png", dpi=300)
    plt.close()

def plot_prob_cdf_across_scales(all_prob_dict, output_dir="figures"):
    """
    all_prob_dict: {level: {scale: [(token, prob), ...]}}
    绘制CDF跨scale对比（2x2子图）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Cumulative Probability Distribution Evolution", fontsize=16, fontweight='bold')
    
    levels = ["Char-EN", "Word-EN", "Char-ZH", "Word-ZH"]
    titles = ["Char-EN CDF", "Word-EN CDF", "Char-ZH CDF", "Word-ZH CDF"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(SCALES)))
    
    for ax, level, ctitle in zip(axes.flatten(), levels, titles):
        for scale, color in zip(SCALES, colors):
            if scale not in all_prob_dict.get(level, {}):
                continue
            sorted_probs = sorted([p for _, p in all_prob_dict[level][scale]], reverse=True)
            cum_probs = np.cumsum(sorted_probs)
            ranks = np.arange(1, len(cum_probs) + 1)
            ax.plot(ranks, cum_probs, marker='o', linestyle='-', color=color,
                    label=f"N={scale//1000 if scale<1_000_000 else scale//1_000_000}M")
        ax.set_xlabel("Rank (Number of Tokens)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(ctitle)
        ax.set_xscale('log')  # log scale for x to show long tail
        ax.legend(title="Scale", loc="lower right")
        ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/prob_cdf_evolution.png", dpi=300)
    plt.close()

def main():
    # 加载语料
    print("Loading corpora...")
    en_texts = load_corpus(EN_CORPUS_PATH)
    zh_texts = load_corpus(ZH_CORPUS_PATH)
    print(f"EN: {len(en_texts)} documents, ZH: {len(zh_texts)} documents")

    # 预分词为 token 流（提高采样效率）
    print("Pre-tokenizing corpora...")
    en_char_tokens = tokenize_en_char(" ".join(en_texts))
    en_word_tokens = tokenize_en_word(" ".join(en_texts))
    zh_char_tokens = tokenize_zh_char("".join(zh_texts))
    zh_word_tokens = tokenize_zh_word("".join(zh_texts))

    token_streams = {
        "Char-EN": en_char_tokens,
        "Word-EN": en_word_tokens,
        "Char-ZH": zh_char_tokens,
        "Word-ZH": zh_word_tokens
    }

    # 收集每种粒度、每个 scale 的前 20 高频项 (for freq plots)
    rank_freq_per_scale = {}   # {scale: {level: [(token, freq), ...] 前20}}
    all_rank_freq = {}        # {level: {scale: [...]}} 用于跨规模对比

    # 收集每种粒度、每个 scale 的前 20 高概率项 (for prob plots)
    prob_per_scale = {}       # {scale: {level: [(token, prob), ...] 前20}}
    all_prob_dict = {}        # {level: {scale: [...]}} 用于CDF跨规模对比

    for scale in SCALES:
        print(f"\n=== Processing scale: {scale:,} tokens ===")
        rank_freq_per_scale[scale] = {}
        prob_per_scale[scale] = {}
        for level, config in ANALYSIS_LEVELS.items():
            print(f"  -> {level}")
            tokens_all = token_streams[level]
            if len(tokens_all) == 0:
                print(f"    Warning: No tokens for {level}")
                continue

            # 采样
            sampled_tokens = sample_tokens(tokens_all, scale)

            # 过滤（Char 级别）
            if "vocab_filter" in config:
                sampled_tokens = [t for t in sampled_tokens if config["vocab_filter"](t)]

            total_n = len(sampled_tokens)
            if total_n == 0:
                continue

            # 频数统计
            counter = Counter(sampled_tokens)

            # 排序频数
            sorted_freq = sorted(counter.items(), key=lambda x: -x[1])

            # 取前20 for plots
            top20_freq = sorted_freq[:20]

            # 概率与熵 (熵用全counter)
            entropy = compute_entropy(counter, total_n)

            # Zipf 拟合 (用前100)
            slope, intercept, r2, _ = fit_zipf(counter)  # 忽略返回的list，因为我们有sorted_freq

            # 新增：计算概率统计
            top_probs, top1_prob, cum_top10, prob_skew = compute_prob_stats(counter, total_n)

            # 更新results
            results.append({
                "Scale": scale,
                "Level": level,
                "VocabSize": len(counter),
                "Entropy": round(entropy, 4),
                "Zipf_Slope": round(slope, 4) if slope else None,
                "Zipf_Intercept": round(intercept, 4) if intercept else None,
                "Zipf_R2": round(r2, 4) if r2 else None,
                "Top1_Prob": round(top1_prob, 6),
                "Cumulative_Top10_Prob": round(cum_top10, 4),
                "Prob_Skewness": round(prob_skew, 4)
            })

            # 存储前20 freq 用于Zipf绘图
            rank_freq_per_scale[scale][level] = top20_freq  # (token, freq)

            # 存储前20 probs 用于prob绘图
            prob_per_scale[scale][level] = top_probs  # (token, prob)

            # 用于跨规模图
            all_rank_freq.setdefault(level, {})[scale] = top20_freq
            all_prob_dict.setdefault(level, {})[scale] = top_probs

        # 每个 scale 画 2×2 Zipf对比图 (用freq)
        plot_zipf_multi(rank_freq_per_scale[scale], scale)

        # 每个 scale 画概率柱状图 (用prob)
        plot_probabilities_multi(prob_per_scale[scale], scale)

    # 全部 scale 完成后画跨规模Zipf演化图 (用freq)
    plot_zipf_across_scales(all_rank_freq)

    # 画CDF演化图 (用prob)
    plot_prob_cdf_across_scales(all_prob_dict)

    # 保存 CSV
    df = pd.DataFrame(results)
    df.to_csv("results/statistics_results.csv", index=False)
    print("\nAll done! Results saved to results/statistics_results.csv")
    print("Zipf plots saved to figures/")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    main()