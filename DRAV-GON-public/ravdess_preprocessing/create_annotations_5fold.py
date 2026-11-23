# -*- coding: utf-8 -*-
"""
按说话人（Actor）划分生成 K 折交叉验证用的标注文件。

用途
----
回应审稿人「问题2：数据集样本量少，需补充 K 折交叉验证」。

为什么必须按 Actor 划分，而不能把所有样本随机打散均分成 K 份
------------------------------------------------------------
RAVDESS 一共 24 个演员，每个演员演同样的 60 条情绪片段。
如果把样本随机打散，同一个演员的片段会同时出现在训练集和测试集里，
模型只要记住「这个人的长相/音色」就能答对，测出来的准确率是虚高的。
这叫**说话人泄漏 (speaker leakage)**，是语音/情感识别领域审稿人会重点挑的问题。
因此必须按演员整体划分：某个演员的所有样本要么都在训练集，要么都在测试集。

本脚本的划分方式（关键设计）
----------------------------
把 24 个演员随机分成 K 组（5 折时为 5/5/5/5/4），第 k 折：

    testing    = 第 k 组
    validation = 第 (k+1) % K 组      ← 复用同一份分区，零额外逻辑
    training   = 其余 K-2 组

为什么还需要 validation：原代码用验证集挑 best checkpoint
（utils.py 的 save_checkpoint 在验证精度刷新时才复制出 _best 文件），
而测试阶段 main.py 要加载这个 _best。若某折没有验证集，测试会直接失败。
让 validation 取「下一组」既保证了与测试集说话人不重叠，又不用再切一次数据。

为什么不能复用仓库里原有的 create_annotations.py
------------------------------------------------
原脚本用 `for i, actor in enumerate(os.listdir(...))` 的**下标 i** 去匹配说话人 id。
而 os.listdir 的返回顺序在不同机器/文件系统上是不保证一致的，
等于换台机器划分结果就变了，且不受随机种子控制 → 实验不可复现。
本脚本改用正则从路径里提取真实的 Actor 编号，并显式设置随机种子。

怎么用
------
    # 生成 5 折（默认读 ravdess_preprocessing/annotations.txt）
    python ravdess_preprocessing/create_annotations_5fold.py

    # 指定 K 与输出前缀
    python ravdess_preprocessing/create_annotations_5fold.py --n_folds 5 \
        --out ravdess_preprocessing/annotations_fold.txt

    # 服务器上的数据换了目录：把旧前缀替换成新前缀
    python ravdess_preprocessing/create_annotations_5fold.py \
        --rebase_from /path/to/dataset --rebase_to /data

脚本会生成 <out 去掉 .txt>_0.txt ... _(K-1).txt，
这与 main.py 里「把 --annotation_path 去掉 .txt 再拼 '_<折号>.txt'」的规则正好对应，
所以训练时这样传即可：

    python main.py --n_folds 5 \
        --annotation_path ravdess_preprocessing/annotations_fold.txt
"""

import argparse
import os
import random
import re
from collections import OrderedDict, defaultdict

# RAVDESS 视频路径形如：
#   /path/to/dataset/RAVDESS/Video_Speech_Actors_01-24_processed/Actor_16/01-01-08-02-01-01-16_facecroppad.npy
# 取其中的 Actor_16 → 16
_ACTOR_RE = re.compile(r'Actor[_-]?(\d+)', re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(
        description='按说话人划分生成 K 折交叉验证标注文件',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', default='ravdess_preprocessing/annotations.txt', type=str,
                   help='原始标注文件（当作全量样本清单使用）')
    p.add_argument('--out', default='ravdess_preprocessing/annotations_fold.txt', type=str,
                   help='输出前缀。实际会生成 <前缀去 .txt>_0.txt ... _(K-1).txt')
    p.add_argument('--n_folds', default=5, type=int, help='折数，审稿人要求 5 折')
    p.add_argument('--seed', default=42, type=int, help='划分随机种子，保证可复现')
    p.add_argument('--rebase_from', default='', type=str,
                   help='需要被替换的旧路径前缀（服务器换目录时用），留空表示不替换')
    p.add_argument('--rebase_to', default='', type=str,
                   help='新的路径前缀')
    p.add_argument('--stratified', action='store_true',
                   help='按标签分层划分（样本级），而非按说话人。用于没有可靠说话人标注、'
                        '且按采集源(video_id)划分会导致折间类别严重失衡的数据集（如 PKU-ER）。'
                        '保证每折各类别比例接近全局，直接回应审稿人 balanced distribution 的要求')
    return p.parse_args()


def extract_speaker(video_path):
    """
    从视频路径里提取说话人标识。

    优先用正则匹配 Actor_16 这样的形式（RAVDESS）；
    匹配不到就退回用「样本所在文件夹名」，并在后面由调用方给出提醒。
    """
    m = _ACTOR_RE.search(video_path)
    if m:
        return 'Actor_{}'.format(int(m.group(1)))
    return os.path.basename(os.path.dirname(video_path))


def rebase(path, old, new):
    """把路径里的旧前缀替换成新前缀。old 为空时原样返回。"""
    if not old:
        return path
    return path.replace(old, new)


def stratified_split(samples, n_folds, seed, out_prefix):
    """
    按标签分层的样本级 K 折划分。

    与按说话人划分的区别：这里每个样本独立分配，同一采集源(video)的样本
    可能同时出现在训练集和测试集。仅当数据集没有可靠的逐样本说话人标注、
    且按采集源划分会导致折间类别严重失衡时使用（PKU-ER 正是这种情况：
    每个 video_id 只含 1~2 种情绪，按 video 分折会出现整类缺失）。

    分层方式：在每个类别内部 shuffle 后轮流分配到 K 折，因此每折的三类
    比例都接近全局分布（PKU-ER 约 288:291:323）。
    第 k 折 testing=第k桶，validation=第(k+1)%K桶，training=其余。
    """
    by_label = defaultdict(list)
    for s in samples:
        by_label[s['label']].append(s)

    rng = random.Random(seed)
    buckets = [[] for _ in range(n_folds)]
    for lab in sorted(by_label):
        items = list(by_label[lab])
        rng.shuffle(items)
        for idx, s in enumerate(items):
            buckets[idx % n_folds].append(s)

    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print('{} 折【分层】交叉验证标注生成（样本级，按标签均衡）'.format(n_folds))
    print('=' * 70)
    print('样本总数 : {}'.format(len(samples)))
    print('类别分布 : {}'.format({k: len(v) for k, v in sorted(by_label.items())}))
    print('随机种子 : {}'.format(seed))

    for k in range(n_folds):
        test_ids = set(id(x) for x in buckets[k])
        val_ids = set(id(x) for x in buckets[(k + 1) % n_folds])
        rows = {'training': [], 'validation': [], 'testing': []}
        for s in samples:
            entry = '{};{};{}'.format(s['video_path'], s['audio_path'], s['label'])
            if id(s) in test_ids:
                rows['testing'].append(entry + ';testing')
            elif id(s) in val_ids:
                rows['validation'].append(entry + ';validation')
            else:
                rows['training'].append(entry + ';training')
        out_path = '{}_{}.txt'.format(out_prefix, k)
        with open(out_path, 'w') as f:
            for subset in ['training', 'validation', 'testing']:
                f.write('\n'.join(rows[subset]))
                f.write('\n')
        from collections import Counter
        ct = Counter(r.split(';')[2] for r in rows['testing'])
        print('第 {} 折 -> {}  (train={} val={} test={} | test类别={})'.format(
            k, out_path, len(rows['training']), len(rows['validation']),
            len(rows['testing']), dict(ct)))


def main():
    args = parse_args()

    if not os.path.isfile(args.src):
        raise SystemExit(
            '找不到标注文件: {}\n'
            '请先用 --src 指向你服务器上的 annotations.txt'.format(args.src))

    # ---------- 1. 读取样本清单 ----------
    samples = []
    with open(args.src, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            if len(parts) < 4:
                raise SystemExit(
                    '第 {} 行格式不对，应为 video_path;audio_path;label;subset\n  实际: {}'
                    .format(lineno, line))
            video_path, audio_path, label = parts[0], parts[1], parts[2]
            samples.append({
                'video_path': rebase(video_path, args.rebase_from, args.rebase_to),
                'audio_path': rebase(audio_path, args.rebase_from, args.rebase_to),
                'label': label,
                'speaker': extract_speaker(video_path),
            })

    # 分层模式：按标签样本级划分，跳过说话人分组（用于无 speaker 标注的 PKU-ER）
    if args.stratified:
        out_prefix = args.out[:-4] if args.out.endswith('.txt') else args.out
        stratified_split(samples, args.n_folds, args.seed, out_prefix)
        return

    # ---------- 2. 按说话人分组 ----------
    by_speaker = OrderedDict()
    for s in samples:
        by_speaker.setdefault(s['speaker'], []).append(s)
    speakers = sorted(by_speaker.keys())

    if args.n_folds > len(speakers):
        raise SystemExit(
            '说话人只有 {} 个，无法分成 {} 折（折数必须 ≤ 说话人数）'
            .format(len(speakers), args.n_folds))

    # 分不清是否是按真人划分时提醒一下，避免悄悄退化成按目录划分
    if not any(_ACTOR_RE.search(k) for k in speakers):
        print('[提醒] 没有在路径里匹配到 Actor 编号，已退回用「文件夹名」当说话人标识。')
        print('       当前说话人标识示例: {}'.format(speakers[:3]))
        print('       请确认这确实等价于按说话人划分，否则会发生说话人泄漏。')

    # ---------- 3. 随机把说话人分成 K 组 ----------
    # 用固定种子打乱，保证同一台机器和不同机器上都能复现同一份划分。
    rng = random.Random(args.seed)
    shuffled = list(speakers)
    rng.shuffle(shuffled)

    groups = [[] for _ in range(args.n_folds)]
    for idx, sp in enumerate(shuffled):
        groups[idx % args.n_folds].append(sp)  # 轮流分配，天然均衡（24 人分 5 折 → 5/5/5/5/4）

    out_prefix = args.out[:-4] if args.out.endswith('.txt') else args.out
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print('{} 折交叉验证标注生成'.format(args.n_folds))
    print('=' * 70)
    print('样本总数 : {}'.format(len(samples)))
    print('说话人数 : {}'.format(len(speakers)))
    print('随机种子 : {}'.format(args.seed))
    print('输出前缀 : {}'.format(out_prefix))

    # ---------- 4. 逐折写出 ----------
    for k in range(args.n_folds):
        # 把分组结果摊平回「说话人 id 列表」，再求差集得到训练说话人
        all_grouped = [sp for g in groups for sp in g]
        test_speakers = set(groups[k])
        val_speakers = set(groups[(k + 1) % args.n_folds])
        train_speakers = set(all_grouped) - test_speakers - val_speakers

        # 三个子集两两不能有交集，否则就是说话人泄漏
        assert not (test_speakers & val_speakers), '第 {} 折：验证集与测试集说话人重叠'.format(k)
        assert not (test_speakers & train_speakers), '第 {} 折：训练集与测试集说话人重叠'.format(k)
        assert not (val_speakers & train_speakers), '第 {} 折：训练集与验证集说话人重叠'.format(k)

        rows = {'training': [], 'validation': [], 'testing': []}
        for sp in all_grouped:
            subset = ('testing' if sp in test_speakers
                      else 'validation' if sp in val_speakers
                      else 'training')
            for s in by_speaker[sp]:
                rows[subset].append('{};{};{};{}'.format(
                    s['video_path'], s['audio_path'], s['label'], subset))

        out_path = '{}_{}.txt'.format(out_prefix, k)
        with open(out_path, 'w') as f:
            for subset in ['training', 'validation', 'testing']:
                f.write('\n'.join(rows[subset]))
                f.write('\n')

        print('\n第 {} 折 -> {}'.format(k, out_path))
        print('    training   {:>4} 条  (说话人 {})'.format(len(rows['training']), len(train_speakers)))
        print('    validation {:>4} 条  (说话人 {})'.format(len(rows['validation']), len(val_speakers)))
        print('    testing    {:>4} 条  (说话人 {})'.format(len(rows['testing']), len(test_speakers)))

    print('\n' + '=' * 70)
    print('全部完成。训练时这样传参：')
    print('    python main.py --n_folds {} --annotation_path {}'.format(args.n_folds, args.out))
    print('=' * 70)


if __name__ == '__main__':
    main()
