# -*- coding: utf-8 -*-
"""
对逐样本预测结果做统计，产出论文需要的表格与结论数字。

用途（回应审稿人问题9）
----------------------
审稿人要求补充「单一模态分类错误，但总体预测稳定」的统计结果：
  1. 统计测试集中音频/视频模态预测不一致的样本占比；
  2. 统计对应错分类型的分布比例。

本脚本读取 robustness_test.py 落盘的逐样本预测 CSV，
计算错分类型分布与三个关键量化指标，并输出 CSV / 控制台表格 / LaTeX 表格。

错分类型体系（C1~C6，互斥且完备）
---------------------------------
设 cm / ca / cv 分别表示融合头 M、音频头 A、视频头 V 是否预测正确（1=正确）：

  C1 三者皆对      cm=1, ca=1, cv=1      冗余一致，最容易的样本
  C2 音频错视频对  cm=1, ca=0, cv=1      视觉补上了音频的错误
  C3 音频对视频错  cm=1, ca=1, cv=0      音频补上了视觉的错误
  C4 双模皆错融合对 cm=1, ca=0, cv=0     ★ 融合最强的证据
  C5 至少一个单模对但融合错  cm=0, (ca=1 或 cv=1)   融合失误
  C6 三者皆错      cm=0, ca=0, cv=0      硬样本

六个类别覆盖了 (cm, ca, cv) 的全部 8 种组合且互不重叠，
因此六个计数之和恒等于样本总数（脚本里有断言保证这一点）。

三个关键量化指标
----------------
1. A/V 不一致率      ：pred_a != pred_v 的样本占比。
                       高说明两个模态确实在看不同的东西，互补有空间。
2. 条件挽救率        ：在「两个单模态预测不一致」的样本里，融合头 M 仍然正确的比例。
                       这是回应审稿人最关键的数字 —— 若它远高于单模准确率，
                       就说明「单模态分错了，但整体预测依然稳定」。
3. 互补增益          ：M 预测正确的样本中，「至少一个单模态头是错的」所占比例
                       = (C2 + C3 + C4) / (cm=1)。
                       说明融合并不是简单复述某个单模态的结果。

怎么用
------
    # 统计 dumps/ 下所有逐样本预测
    python analyze_stats.py

    # 指定目录与输出
    python analyze_stats.py --dumps 'dumps/*.csv' --out_csv q9_stats.csv

    # 同时产出 LaTeX 表格
    python analyze_stats.py --latex_dir tables

    # 不读数据，只跑统计逻辑自检（本机即可运行，用来验证分类体系自洽）
    python analyze_stats.py --self_test
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

# 六个错分类型的顺序与中文说明
CATEGORIES = [
    ('C1', '三者皆对', 'cm=1, ca=1, cv=1'),
    ('C2', '音频错视频对、融合对', 'cm=1, ca=0, cv=1'),
    ('C3', '音频对视频错、融合对', 'cm=1, ca=1, cv=0'),
    ('C4', '双模皆错但融合对', 'cm=1, ca=0, cv=0'),
    ('C5', '至少一个单模对但融合错', 'cm=0, ca=1 或 cv=1'),
    ('C6', '三者皆错', 'cm=0, ca=0, cv=0'),
]


def add_derived_columns(df):
    """
    给逐样本表补上派生列。

    只放与「对错」相关的布尔列，不做任何聚合，
    这样同一份数据可以反复用不同口径统计。
    """
    df = df.copy()
    df['cm'] = (df['pred_m'] == df['label']).astype(int)
    df['ca'] = (df['pred_a'] == df['label']).astype(int)
    df['cv'] = (df['pred_v'] == df['label']).astype(int)
    # 音频头与视频头给出不同预测 → 两个模态产生了分歧
    df['disagree'] = (df['pred_a'] != df['pred_v']).astype(int)
    df['category'] = df.apply(_classify_row, axis=1)
    return df


def _classify_row(r):
    """
    把一条样本归到 C1~C6。

    这里刻意写成「显式列举 + 兜底抛错」而不是层层 if-else：
    一旦六个分支没能覆盖全部组合，会立刻抛异常而不是静默漏统计。
    """
    cm, ca, cv = int(r['cm']), int(r['ca']), int(r['cv'])

    if cm == 1 and ca == 1 and cv == 1:
        return 'C1'
    if cm == 1 and ca == 0 and cv == 1:
        return 'C2'
    if cm == 1 and ca == 1 and cv == 0:
        return 'C3'
    if cm == 1 and ca == 0 and cv == 0:
        return 'C4'
    if cm == 0 and (ca == 1 or cv == 1):
        return 'C5'
    if cm == 0 and ca == 0 and cv == 0:
        return 'C6'
    raise ValueError('无法归类的样本: cm={}, ca={}, cv={}'.format(cm, ca, cv))


def extract_actor(path_series):
    """
    从样本路径里提取演员编号，便于按说话人做二次统计。
    取不到时返回空字符串，不影响主流程。
    """
    def _one(p):
        if not isinstance(p, str):
            return ''
        m = re.search(r'Actor[_-]?(\d+)', p)
        return m.group(1) if m else ''
    return path_series.map(_one)


def summarize_by_snr(df):
    """
    按信噪比分组统计，得到每个噪声强度下的完整画像。

    返回 DataFrame，每行一个信噪比档位。
    由于 dump 里带 fold 列，这里会把同一信噪比下**所有折**的样本合并统计，
    这正是交叉验证该有的汇总方式（而不是先对各折取平均再平均）。
    """
    rows = []
    # snr_db 可能是字符串 'clean' / '10' / '-5'，按出现顺序排序：
    # clean 排最前（噪声最弱），其余按数值从大到小
    def _snr_key(v):
        if str(v).lower() == 'clean':
            return float('inf')
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('-inf')

    for snr, g in df.groupby('snr_db', sort=False):
        n = len(g)
        row = {
            'snr_db': snr,
            'n_samples': n,
            'acc_m': 100.0 * g['cm'].mean(),
            'acc_a': 100.0 * g['ca'].mean(),
            'acc_v': 100.0 * g['cv'].mean(),
            'disagree_rate': 100.0 * g['disagree'].mean(),
        }
        # 各类别的数量与占比
        counts = g['category'].value_counts()
        for code, _, _ in CATEGORIES:
            c = int(counts.get(code, 0))
            row[code + '_count'] = c
            row[code + '_pct'] = 100.0 * c / n if n else 0.0

        # 条件挽救率：不一致样本里融合头仍正确的比例
        dis = g[g['disagree'] == 1]
        row['rescue_rate'] = 100.0 * dis['cm'].mean() if len(dis) else float('nan')

        # 把融合准确率拆成「两模态一致」与「两模态不一致」两个子集。
        # 这两个数字放在一起对比，就是「单模态有分歧时融合是否依然稳定」的直观证据。
        agr = g[g['disagree'] == 0]
        row['acc_m_agree'] = 100.0 * agr['cm'].mean() if len(agr) else float('nan')
        row['acc_m_disagree'] = row['rescue_rate']

        # 互补增益：融合正确的样本中，至少一个单模态头是错的比例
        ok = g[g['cm'] == 1]
        if len(ok):
            at_least_one_wrong = ((ok['ca'] == 0) | (ok['cv'] == 0)).mean()
            row['complement_gain'] = 100.0 * at_least_one_wrong
        else:
            row['complement_gain'] = float('nan')

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values('snr_db', key=lambda s: s.map(_snr_key), ascending=False).reset_index(drop=True)


def load_dumps(pattern):
    """读取一个或多个逐样本预测 CSV 并合并。"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(
            '没有找到任何逐样本预测文件: {}\n'
            '请先运行 robustness_test.py 生成 dumps/'.format(pattern))
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d['source_file'] = os.path.basename(f)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print('读入 {} 个文件，共 {} 条样本'.format(len(files), len(df)))
    return df


def write_latex(summary, out_dir):
    """输出两张可直接粘进论文的 LaTeX 表格。"""
    os.makedirs(out_dir, exist_ok=True)

    # 表 1：各信噪比下三个头的准确率
    p1 = os.path.join(out_dir, 'table_snr_accuracy.tex')
    with open(p1, 'w') as f:
        f.write('% 各信噪比下融合头/音频头/视频头的准确率(%%)\n')
        f.write('\\begin{tabular}{lrrr}\n\\hline\n')
        f.write('SNR (dB) & 融合 M & 音频 A & 视频 V \\\\\n\\hline\n')
        for _, r in summary.iterrows():
            f.write('{} & {:.2f} & {:.2f} & {:.2f} \\\\\n'.format(
                r['snr_db'], r['acc_m'], r['acc_a'], r['acc_v']))
        f.write('\\hline\n\\end{tabular}\n')

    # 表 2：错分类型分布
    p2 = os.path.join(out_dir, 'table_error_types.tex')
    with open(p2, 'w') as f:
        f.write('% 错分类型分布(%%)\n')
        f.write('\\begin{tabular}{l' + 'r' * len(CATEGORIES) + '}\n\\hline\n')
        f.write('SNR (dB) & ' + ' & '.join(c[0] for c in CATEGORIES) + ' \\\\\n\\hline\n')
        for _, r in summary.iterrows():
            f.write('{} & '.format(r['snr_db'])
                    + ' & '.join('{:.2f}'.format(r[c[0] + '_pct']) for c in CATEGORIES)
                    + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')

    print('LaTeX 表格已写入: {}, {}'.format(p1, p2))


def print_report(df, summary):
    """在控制台打印一份人能直接读的统计报告。"""
    print('\n' + '=' * 78)
    print('审稿人问题9：单模态不一致与错分类型统计')
    print('=' * 78)

    # 只看干净测试集的结论（问题9 针对的是正常场景）
    clean = summary[summary['snr_db'].astype(str).str.lower() == 'clean']
    src = clean.iloc[0] if len(clean) else summary.iloc[0]

    print('\n【干净测试集关键结论】样本数 {}'.format(int(src['n_samples'])))
    print('  融合头 M 准确率      : {:.2f}%'.format(src['acc_m']))
    print('  音频头 A 准确率      : {:.2f}%'.format(src['acc_a']))
    print('  视频头 V 准确率      : {:.2f}%'.format(src['acc_v']))
    print('  A/V 预测不一致率     : {:.2f}%'.format(src['disagree_rate']))
    print('  ★ 条件挽救率         : {:.2f}%  (不一致的样本中融合头仍正确的比例)'.format(src['rescue_rate']))
    print('    两模态一致时的 M   : {:.2f}%'.format(src['acc_m_agree']))
    print('    两模态不一致时的 M : {:.2f}%  (与上一行对比即可看出融合是否稳定)'.format(src['acc_m_disagree']))
    print('  ★ 互补增益           : {:.2f}%  (融合正确的样本中至少一个单模态错的比例)'.format(
        src['complement_gain']))

    print('\n【错分类型分布】(占全部测试样本的百分比)')
    print('  {:<4s} {:<24s} {:>8s} {:>8s}'.format('类别', '含义', '数量', '占比%'))
    for code, desc, _ in CATEGORIES:
        print('  {:<4s} {:<24s} {:>8d} {:>8.2f}'.format(
            code, desc, int(src[code + '_count']), src[code + '_pct']))

    print('\n【各信噪比下的表现】')
    cols = ['snr_db', 'n_samples', 'acc_m', 'acc_a', 'acc_v', 'disagree_rate',
            'rescue_rate', 'complement_gain']
    tmp = summary[cols].copy()
    tmp.columns = ['SNR', 'N', 'M%', 'A%', 'V%', '不一致%', '挽救%', '互补%']
    print(tmp.to_string(index=False, float_format=lambda x: '{:.2f}'.format(x)))

    # 直接给出可用于论文的对比结论：噪声下 A 掉多少、M 掉多少
    if len(clean) and len(summary) > 1:
        worst = summary.iloc[-1]
        drop_a = src['acc_a'] - worst['acc_a']
        drop_m = src['acc_m'] - worst['acc_m']
        print('\n【视觉模态的辅助作用】从 clean 到 {} dB：'.format(worst['snr_db']))
        print('  纯音频头 A 下降 : {:.2f} 个百分点'.format(drop_a))
        print('  融合头   M 下降 : {:.2f} 个百分点'.format(drop_m))
        if drop_a > 0:
            print('  → 融合比纯音频少下降 {:.1f}%，视觉模态确实起到了抗噪兜底作用'.format(
                100.0 * (drop_a - drop_m) / drop_a))

    print('\n' + '=' * 78)


def self_test():
    """
    统计逻辑自检：用合成的预测结果验证分类体系自洽。
    无需任何真实数据，本机直接 `python analyze_stats.py --self_test` 即可运行。
    """
    print('统计逻辑自检')
    rng = np.random.default_rng(0)
    n = 1000
    label = rng.integers(0, 8, size=n)

    # 造三个头的预测：让它们有一定概率正确、彼此有一定概率不一致
    def _pred(acc):
        p = label.copy()
        flip = rng.random(n) > acc
        p[flip] = (p[flip] + rng.integers(1, 8, size=flip.sum())) % 8
        return p

    df = pd.DataFrame({
        'label': label,
        'pred_m': _pred(0.85),
        'pred_a': _pred(0.70),
        'pred_v': _pred(0.65),
        'snr_db': 'clean',
    })
    df = add_derived_columns(df)

    failures = []

    # 1) 六类计数之和 == 样本总数
    counts = df['category'].value_counts()
    total = sum(int(counts.get(c[0], 0)) for c in CATEGORIES)
    print('  六类计数之和 = {}，样本总数 = {}'.format(total, n))
    if total != n:
        failures.append('六类计数之和({}) != 样本总数({})'.format(total, n))

    # 2) 每个类别的定义都成立（互斥性靠定义保证，这里验证归类没错配）
    for code, desc, _ in CATEGORIES:
        sub = df[df['category'] == code]
        if len(sub) == 0:
            continue
        if code == 'C1' and not ((sub['cm'] == 1) & (sub['ca'] == 1) & (sub['cv'] == 1)).all():
            failures.append('C1 定义不成立')
        if code == 'C4' and not ((sub['cm'] == 1) & (sub['ca'] == 0) & (sub['cv'] == 0)).all():
            failures.append('C4 定义不成立')
        if code == 'C6' and not ((sub['cm'] == 0) & (sub['ca'] == 0) & (sub['cv'] == 0)).all():
            failures.append('C6 定义不成立')

    # 3) C2+C3+C4 应等于「融合正确且至少一个单模态错」的样本数
    lhs = int((df['category'].isin(['C2', 'C3', 'C4'])).sum())
    rhs = int(((df['cm'] == 1) & ((df['ca'] == 0) | (df['cv'] == 0))).sum())
    print('  C2+C3+C4 = {}，融合正确且至少一个单模错 = {}'.format(lhs, rhs))
    if lhs != rhs:
        failures.append('C2+C3+C4 与互补增益的分子不一致')

    # 4) 各类别占比之和应为 100%
    summary = summarize_by_snr(df)
    pct_sum = sum(summary.iloc[0][c[0] + '_pct'] for c in CATEGORIES)
    print('  各类别占比之和 = {:.4f}%'.format(pct_sum))
    if abs(pct_sum - 100.0) > 1e-6:
        failures.append('占比之和不为 100%')

    # 5) 准确率应与直接计算一致
    if abs(summary.iloc[0]['acc_m'] - 100.0 * (df['pred_m'] == df['label']).mean()) > 1e-9:
        failures.append('M 准确率统计不一致')

    print('\n' + '=' * 60)
    if failures:
        print('自检未通过：')
        for f in failures:
            print('  - ' + f)
        return 1
    print('自检全部通过（分类体系互斥且完备，指标自洽）')
    return 0


def main():
    p = argparse.ArgumentParser(
        description='逐样本预测统计：错分类型分布与单模态不一致分析',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dumps', default='dumps/*.csv', type=str,
                   help='逐样本预测 CSV 的通配路径')
    p.add_argument('--out_csv', default='q9_stats.csv', type=str,
                   help='统计结果输出 CSV')
    p.add_argument('--latex_dir', default='', type=str,
                   help='非空时额外输出 LaTeX 表格到该目录')
    p.add_argument('--self_test', action='store_true',
                   help='不读数据，只跑统计逻辑自检')
    args = p.parse_args()

    if args.self_test:
        import sys
        sys.exit(self_test())

    df = load_dumps(args.dumps)
    df = add_derived_columns(df)
    summary = summarize_by_snr(df)

    # 自洽性断言：真实数据上同样必须成立，不成立说明脚本或输入有问题
    total = sum(int(summary[c[0] + '_count'].sum()) for c in CATEGORIES)
    assert total == len(df), \
        '错分类型计数之和({}) 与样本总数({}) 不一致，请检查输入'.format(total, len(df))

    summary.to_csv(args.out_csv, index=False, float_format='%.4f')
    print('统计结果已写入: {}'.format(args.out_csv))

    print_report(df, summary)

    if args.latex_dir:
        write_latex(summary, args.latex_dir)


if __name__ == '__main__':
    main()
