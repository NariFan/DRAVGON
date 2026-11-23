# -*- coding: utf-8 -*-
"""
把 BEATs 音频特征预先算好存盘，供训练时直接读取。

为什么需要它
------------
datasets/ravdess.py 每取一个样本都要在 CPU 上跑一次 BEATs（12 层 transformer）。
实测 PKU-ER 上「等数据」占单 batch 耗时的 85%（1.65s / 1.93s），
四组实验全部跑完要 12 小时以上。BEATs 是**冻结的**，对同一条音频永远输出同一个结果，
所以完全可以只算一次。

本脚本用 GPU 把标注文件里所有音频的 BEATs 特征一次性算完存成 .pt，
算完 902 条约半分钟；训练时 dataset 直接读文件，省掉重复推理。

重要前提（不要乱用缓存）
------------------------
缓存只对**不加噪**的音频有效。审稿人问题3 的噪声实验是在波形层加噪之后再提特征
（见 noise_utils.py），噪声必须真的进入 BEATs，否则实验就失去意义。
因此 datasets/ravdess.py 只在 snr is None 时读缓存，加噪场景一律走在线计算。

怎么用
------
    python extract_beats_cache.py \
        --annotation PKU_ER_preprocessing/annotations_pku.txt \
        --out_dir data/cache/beats

训练时通过 --beats_cache_dir 指到同一个目录即可；不传该参数行为与以前完全一致。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.ravdess import load_audio, RAVDESS  # noqa: E402
from utils import safe_torch_load  # noqa: E402


def cache_path_for(audio_path, out_dir):
    """音频绝对路径 -> 缓存文件路径。把路径里的分隔符换成 __，保证唯一。"""
    key = audio_path.replace('/', '__').replace('\\', '__')
    if key.endswith('.wav'):
        key = key[:-4]
    return os.path.join(out_dir, key + '.pt')


def main():
    p = argparse.ArgumentParser(description='预提取 BEATs 特征缓存')
    p.add_argument('--annotation', required=True,
                   help='标注文件（可传多个，用逗号分隔）')
    p.add_argument('--out_dir', required=True, help='缓存输出目录')
    p.add_argument('--beats_ckpt', default='./pre-trained_checkpoints/BEATs_iter3.pt')
    p.add_argument('--backbone', default='beats', choices=['beats', 'emotion2vec'],
                   help='音频基座。emotion2vec 用于回应审稿人 C1 的基座对照实验')
    p.add_argument('--emotion2vec_dir', default='./pre-trained_checkpoints/emotion2vec',
                   help='emotion2vec 权重目录（--backbone emotion2vec 时生效）')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # ---------- 收集所有音频路径 ----------
    audio_paths = []
    for ann in args.annotation.split(','):
        ann = ann.strip()
        if not ann:
            continue
        with open(ann) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(';')
                if len(parts) < 2:
                    continue
                ap = parts[1]
                if ap not in audio_paths:
                    audio_paths.append(ap)

    missing = [a for a in audio_paths if not os.path.exists(a)]
    if missing:
        raise SystemExit('有 {} 条音频文件不存在，例如: {}'.format(len(missing), missing[0]))

    os.makedirs(args.out_dir, exist_ok=True)
    print('音频总数 : {}'.format(len(audio_paths)))
    print('缓存目录 : {}'.format(args.out_dir))

    # ---------- 加载基座模型 ----------
    if args.backbone == 'emotion2vec':
        from emotion2vec_feat import load_emotion2vec_model
        model = load_emotion2vec_model(args.emotion2vec_dir, device=args.device)
        print('emotion2vec 已加载到', args.device)
    else:
        from beats.BEATs import BEATs, BEATsConfig
        ckpt = safe_torch_load(args.beats_ckpt)
        model = BEATs(BEATsConfig(ckpt['cfg']))
        model.load_state_dict(ckpt['model'])
        model.eval().to(args.device)
        print('BEATs 已加载到', args.device)

    # ---------- 与 datasets/ravdess.py 保持一致的在线计算 ----------
    from datasets.ravdess import get_mfccs, get_cqt
    import librosa

    # 特征提取函数：根据基座选择
    def extract_backbone_feature(y22):
        y16 = librosa.resample(y22, orig_sr=22050, target_sr=16000)
        if args.backbone == 'emotion2vec':
            from emotion2vec_feat import extract_emotion2vec_features
            return extract_emotion2vec_features(model, y16, device=args.device)
        t = torch.from_numpy(y16).float().unsqueeze(0).to(args.device)
        pm = torch.zeros(t.shape, dtype=torch.bool, device=t.device)
        with torch.no_grad():
            rep = model.extract_features(t, padding_mask=pm)[0]
        return rep.squeeze()[:156, :].transpose(0, 1).cpu()  # (768, 156)

    done = 0
    skipped = 0
    for i, ap in enumerate(audio_paths, 1):
        out = cache_path_for(ap, args.out_dir)
        if os.path.exists(out):
            skipped += 1
            continue

        # 音频特征链路与 datasets/ravdess.py::__getitem__ 严格一致：
        #   22050 Hz 读入 -> MFCC/CQT 直接在 22050 上算 -> 基座特征重采样到 16000 再算
        # 注意 MFCC/CQT 只有不加噪时才能缓存；噪声实验（--snr）会走在线重算。
        y, sr = load_audio(ap, sr=22050)
        mfcc = get_mfccs(y, sr)
        cqt = get_cqt(y, sr)
        backbone_feat = extract_backbone_feature(y)

        torch.save({'mfcc': mfcc, 'cqt': cqt, 'beats': backbone_feat}, out)

        done += 1
        if i % 100 == 0 or i == len(audio_paths):
            print('  {}/{}  新算 {}  已存在 {}'.format(i, len(audio_paths), done, skipped))

    print('完成。新算 {} 条，跳过 {} 条'.format(done, skipped))


if __name__ == '__main__':
    main()
