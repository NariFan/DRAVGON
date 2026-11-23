# -*- coding: utf-8 -*-
"""
RAVDESS 消融实验总表（回应审稿人「Table A1 只做了单折、且组件被联合去掉」）。

设计要点
--------
1. 一「格」= 一个消融配置，用 ABLATIONS 里的一个字典描述；
   实验名由 utils.build_store_name 从配置推导，所以名字里必然带全部消融轴，
   不会出现「看名字不知道跑了什么」的情况。
2. 所有配置共享同一套协议（30 帧 + emotion2vec+ + actor-wise 5 折），
   只有被消融的那一个轴在变，其余全部锁死。
3. 只跑干净测试（--test_only_both），省掉 6/7 的测试时间。

用法
----
    python run_abl_ravdess.py --list                 # 打印全部配置
    python run_abl_ravdess.py --dry_run              # 只打印命令，不执行
    python run_abl_ravdess.py --panel 1              # 只跑 Panel 1
    python run_abl_ravdess.py --id P1-2              # 只跑某一格
    python run_abl_ravdess.py --id P1-2 --folds 0,1  # 只跑某几折
    python run_abl_ravdess.py --panel 1 --epochs 2 --folds 0   # 冒烟测试
"""

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# 协议常量：与论文 RAVDESS 主数字（91.08+-4.55）完全一致
# ---------------------------------------------------------------------------
CONFIG = dict(
    dataset='RAVDESS',
    annotation=os.environ.get('DRAV_ANNOTATION', 'ravdess_preprocessing/annotations_fold30f.txt'),
    frames=30,
    backbone='emotion2vec',
    heads=1,
    n_folds=5,
    n_classes=8,
    batch=4,
    epochs=100,
    lr=0.04,
    result_path=os.environ.get('DRAV_RESULT_PATH', 'results_abl_rav30'),
    beats_cache=os.environ.get('DRAV_BEATS_CACHE', ''),
    summary_csv=os.environ.get('DRAV_SUMMARY_CSV', 'abl_rav30.csv'),
    device=os.environ.get('DRAV_DEVICE', 'cuda'),
    threads=int(os.environ.get('DRAV_THREADS', '8')),
)

# 固定 NIT（noise）作为默认训练方案；w/o DIP 的那几格单独改成 nodropout
DEFAULT_DIP = 'noise'


# ---------------------------------------------------------------------------
# 消融配置总表
#   fusion     : concat | it | gs_it
#   har        : full | deep | handcrafted
#   cross_dir  : both | av | va
#   cross_depth: 单向堆叠层数（容量匹配）
#   mask       : noise(NIT) | softhard(HM) | nodropout(ODT)
# ---------------------------------------------------------------------------
ABLATIONS = [
    # ---------------- Panel 1：G²AF × HAR-Learner 因子设计 ----------------
    dict(id='P1-1', panel=1, fusion='gs_it', har='full', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='完整参考配置（本批独立重跑，不与主表重复计数）'),
    dict(id='P1-2', panel=1, fusion='gs_it', har='deep', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='w/o HAR-Learner：只留 emotion2vec 深度特征'),
    dict(id='P1-3', panel=1, fusion='it', har='full', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='w/o Gumbel 采样融合：交叉注意力保留，改朴素拼接'),
    dict(id='P1-4', panel=1, fusion='it', har='deep', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='w/o G²AF & HAR-Learner（旧表同名行）'),
    dict(id='P1-5', panel=1, fusion='concat', har='full', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='w/o 整个 G²AF：连交叉注意力一起去掉'),
    dict(id='P1-6', panel=1, fusion='concat', har='deep', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='双去掉下界'),

    # ---------------- Panel 2：DIP 训练方案 ----------------
    dict(id='P2-1', panel=2, fusion='gs_it', har='full', cross_dir='both', cross_depth=1,
         mask='nodropout',
         note='w/o DIP（ODT，原始训练）'),
    dict(id='P2-2', panel=2, fusion='gs_it', har='full', cross_dir='both', cross_depth=1,
         mask='noise', alias='P1-1',
         note='NIT（默认方案，配置与 P1-1 完全相同，不重复跑）'),
    dict(id='P2-3', panel=2, fusion='gs_it', har='full', cross_dir='both', cross_depth=1,
         mask='softhard',
         note='HM（混合掩码）'),

    # ---------------- Panel 3：单向 vs 双向（容量匹配） ----------------
    # gs_it 实测：av 单层 115.8k、va 单层 41.7k、双向合计 157.5k
    dict(id='P3-0', panel=3, fusion='gs_it', har='full', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP, alias='P1-1',
         note='双向基线（配置与 P1-1 完全相同，157.5k，不重复跑）'),
    dict(id='P3-1', panel=3, fusion='gs_it', har='full', cross_dir='av', cross_depth=1,
         mask=DEFAULT_DIP,
         note='单向 音频<-视觉（115.8k，朴素版）'),
    dict(id='P3-2', panel=3, fusion='gs_it', har='full', cross_dir='av', cross_depth=2,
         mask=DEFAULT_DIP,
         note='单向 音频<-视觉 ×2 层（231.6k，容量反超）'),
    dict(id='P3-3', panel=3, fusion='gs_it', har='full', cross_dir='va', cross_depth=1,
         mask=DEFAULT_DIP,
         note='单向 视觉<-音频（41.7k，朴素版）'),
    dict(id='P3-4', panel=3, fusion='gs_it', har='full', cross_dir='va', cross_depth=4,
         mask=DEFAULT_DIP,
         note='单向 视觉<-音频 ×4 层（166.8k，容量反超）'),

    # ---------------- Panel 4：可选补充 ----------------
    dict(id='P4-1', panel=4, fusion='gs_it', har='handcrafted', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='只留 MFCC+CQT，去掉深度特征（反向对照）'),
    dict(id='P4-2', panel=4, fusion='it', har='handcrafted', cross_dir='both', cross_depth=1,
         mask=DEFAULT_DIP,
         note='同上 + w/o Gumbel'),
    dict(id='P4-3', panel=4, fusion='gs_it', har='full', cross_dir='av', cross_depth=1,
         mask='nodropout',
         note='单向 × ODT：确认对称收益不依赖 DIP'),
    dict(id='P4-4', panel=4, fusion='gs_it', har='full', cross_dir='va', cross_depth=1,
         mask='nodropout',
         note='单向 × ODT（另一方向）'),
]


def build_cmd(cfg, folds=None, epochs=None, result_path=None, folds_total=None, threads=None):
    cmd = [
        sys.executable, 'main.py',
        '--dataset', CONFIG['dataset'],
        '--annotation_path', CONFIG['annotation'],
        '--n_folds', str(folds_total or CONFIG['n_folds']),
        '--n_classes', str(CONFIG['n_classes']),
        '--sample_duration', str(CONFIG['frames']),
        '--audio_backbone', CONFIG['backbone'],
        '--fusion', cfg['fusion'],
        '--num_heads', str(CONFIG['heads']),
        '--har', cfg['har'],
        '--cross_dir', cfg['cross_dir'],
        '--cross_depth', str(cfg['cross_depth']),
        '--mask', cfg['mask'],
        '--batch_size', str(CONFIG['batch']),
        '--n_epochs', str(epochs or CONFIG['epochs']),
        '--learning_rate', str(CONFIG['lr']),
        '--device', CONFIG['device'],
        '--n_threads', str(threads or CONFIG['threads']),
        '--result_path', result_path or CONFIG['result_path'],
        # 冒烟/单格调试时结果目录被覆盖，summary CSV 也一并隔离，避免污染正式 CSV。
        '--summary_csv', (result_path + '.csv') if result_path else CONFIG['summary_csv'],
        '--test_only_both',
        '--tag', cfg['id'],
    ]
    if CONFIG['beats_cache']:
        cmd += ['--beats_cache_dir', CONFIG['beats_cache']]
    if folds:
        cmd += ['--fold_list', folds]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', type=int, default=0, help='只跑某个 panel，0=全部')
    p.add_argument('--id', default='', help='只跑某一格，如 P1-2')
    p.add_argument('--folds', default='', help='只跑某些折，如 0,1,2')
    p.add_argument('--summary_csv', default='', help='覆盖汇总 CSV 路径')
    p.add_argument('--epochs', type=int, default=0, help='覆盖 epoch 数（冒烟测试用）')
    p.add_argument('--result_path', default='', help='覆盖结果根目录（冒烟测试用）')
    p.add_argument('--folds_total', type=int, default=0, help='覆盖折数（冒烟测试用）')
    p.add_argument('--threads', type=int, default=0, help='覆盖 DataLoader worker 数')
    p.add_argument('--list', action='store_true', help='只打印配置表')
    p.add_argument('--dry_run', action='store_true', help='只打印命令不执行')
    args = p.parse_args()

    cfgs = ABLATIONS
    if args.panel:
        cfgs = [c for c in cfgs if c['panel'] == args.panel]
    if args.id:
        # 显式指定某一格时，即使是别名也照跑（方便单独复现）
        cfgs = [c for c in cfgs if c['id'] == args.id]
    else:
        # 别名格与它的参照格配置完全一样，默认跳过，避免重复训练
        skipped = [c for c in cfgs if c.get('alias')]
        cfgs = [c for c in cfgs if not c.get('alias')]
        for c in skipped:
            print('[skip] %s == %s（配置完全相同，不重复训练）' % (c['id'], c['alias']))
    if not cfgs:
        raise SystemExit('没有匹配的配置')

    if args.list:
        print('%-6s %-6s %-8s %-13s %-4s %-2s %-10s  %s' %
              ('ID', 'PANEL', 'FUSION', 'HAR', 'DIR', 'D', 'MASK', 'NOTE'))
        for c in cfgs:
            note = c['note']
            if c.get('alias'):
                note = '≡ %s | %s' % (c['alias'], note)
            print('%-6s %-6d %-8s %-13s %-4s %-2d %-10s  %s' %
                  (c['id'], c['panel'], c['fusion'], c['har'],
                   c['cross_dir'], c['cross_depth'], c['mask'], note))
        return

    for c in cfgs:
        cmd = build_cmd(c, folds=args.folds, epochs=args.epochs or None,
                        result_path=args.result_path or None,
                        folds_total=args.folds_total or None,
                        threads=args.threads or None)
        if args.summary_csv:
            cmd = cmd[:cmd.index('--summary_csv')] + ['--summary_csv', args.summary_csv] + cmd[cmd.index('--summary_csv') + 2:]
        print('\n' + '=' * 100)
        print('[%s] %s' % (c['id'], c['note']))
        print('=' * 100)
        print(' '.join(cmd))
        if args.dry_run:
            continue
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        rc = subprocess.call(cmd)
        if rc != 0:
            print('[FAIL] %s 退出码 %d' % (c['id'], rc))
            return rc
    return 0


if __name__ == '__main__':
    sys.exit(main())
