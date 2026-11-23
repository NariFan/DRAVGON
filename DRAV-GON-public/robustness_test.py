# -*- coding: utf-8 -*-
"""
在测试集上按多梯度信噪比 (SNR) 加噪，定量评估模型抗干扰能力。

用途（一次运行同时产出两组审稿人所需数据）
------------------------------------------
- 问题3：在测试集上添加不同 dB 的加性噪声，统计「融合头 M / 音频头 A / 视频头 V」
  在各噪声强度下的准确率，对比性能下降幅度。
  由于 A 头是纯音频通道，随着噪声变强它的准确率应当快速下跌，
  而融合头 M 因为有视觉信息兜底，下跌幅度应明显更小 ——
  这条对比曲线本身就是「视觉模态在噪声下起到辅助作用」的直接证据。
- 问题9：把每条测试样本的 (真实标签, M/A/V 三个头的预测) 逐条落盘。
  其中 SNR=clean（不加噪）那一份就是「单模态分类错误但总体预测稳定」
  统计分析的数据源，因此不需要为此再单独跑一遍。

为什么单独写一个脚本而不是改 main.py
------------------------------------
main.py 的测试段固定跑 7 个「模态破坏」组合，而且往 test_set_bestval<fold>.txt
追加内容。若在里面加 SNR 循环，测试量会变成 折数 × 7 × SNR档数，
同时会污染已有的结果文件。本脚本保持独立，不改动任何已有行为。

怎么用（在 DRAV-GON/ 目录下执行）
----------------------------------
    # 对 5 折的全部 best checkpoint 跑 7 个噪声档
    python robustness_test.py \
        --result_path results_q2 --store_name gs_ia-1-noise \
        --n_folds 5 \
        --annotation_path ravdess_preprocessing/annotations_fold.txt

    # 只想先试一折、两个噪声档，确认链路通
    python robustness_test.py --result_path results_q2 --store_name gs_ia-1-noise \
        --folds 0 --snr_list clean,0

产物
----
    dumps/<store_name>_fold<k>_snr<tag>.csv   逐样本预测（问题9 用）
    q3_snr.csv                                各 (折, 信噪比) 的准确率汇总（问题3 用）
    <result_path>/robust_fold<k>.log          与原来同格式的运行日志

注意
----
噪声是在**波形层**加的：读取 wav 之后、提取 MFCC/CQT/BEATs 之前，
因此三种声学特征会一起被污染，这才是真正的声学噪声。详见 noise_utils.py。
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from dataset import get_test_set
import transforms
from utils import Logger, safe_torch_load
from validation import val_epoch


def parse_extra_args():
    """
    解析本脚本独有的参数。

    做法上的一个小技巧：先用 parse_known_args 把本脚本的专属参数取走，
    剩下的参数再交给主程序的 parse_opts() 解析。
    这样既能拿到 --fusion / --n_classes 等标准参数（且默认值与 main.py 完全一致），
    又不会让两边 argparse 互相报「无法识别的参数」。
    """
    p = argparse.ArgumentParser(description='多梯度信噪比鲁棒性测试')
    p.add_argument('--folds', default='', type=str,
                   help='要测试的折号，逗号分隔，例如 0,1,2,3,4。留空表示全部折')
    p.add_argument('--snr_list', default='clean,20,15,10,5,0,-5', type=str,
                   help='信噪比档位，逗号分隔。clean 表示不加噪；数值单位 dB，越小噪声越强')
    p.add_argument('--dump_dir', default='dumps', type=str,
                   help='逐样本预测的输出目录')
    p.add_argument('--snr_summary_csv', default='q3_snr.csv', type=str,
                   help='问题3 的准确率汇总 CSV 路径（追加写）')
    p.add_argument('--n_threads', dest='n_threads_extra', default=4, type=int,
                   help='数据加载线程数。每折每个档位都会重建数据集，'
                        '线程开太大容易导致显存/内存占用翻倍，默认 4')
    extra, remaining = p.parse_known_args()

    # 把剩余参数交还给主程序的参数解析器
    sys.argv = [sys.argv[0]] + remaining
    opt = parse_opts()
    return opt, extra


def parse_snr_list(text):
    """
    把 'clean,20,15,10,5,0,-5' 解析成 [None, 20.0, 15.0, ...]。
    None 代表干净测试集（不加噪）。
    """
    out = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if item.lower() == 'clean':
            out.append(None)
        else:
            try:
                out.append(float(item))
            except ValueError:
                raise SystemExit('无法解析信噪比档位: {!r}，应为 clean 或数字(dB)'.format(item))
    return out


def snr_tag(snr):
    """把信噪比转成文件名里用的标签：None→clean，10.0→10，-5.0→-5。"""
    if snr is None:
        return 'clean'
    return str(int(snr)) if float(snr).is_integer() else str(snr)


def main():
    opt, extra = parse_extra_args()

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备:', opt.device)

    # 数据加载线程：本脚本会比平时更频繁地重建数据集，默认调小一点更稳妥
    opt.n_threads = extra.n_threads_extra

    # 与 main.py 保持一致的 store_name 拼接（main.py:92-95 会把
    # arch / dataset / sample_duration 拼进最终用于保存权重的 store_name）。
    # 之前本脚本直接用未拼接的 store_name 找权重，导致 5 折训练出的
    # <store>_model:..._RAVDESS_15_best<fold>.pth 全部对不上号、q3/q9 跑空。
    # 只在尚未拼接时才拼（以 duration 结尾视为已拼），避免重复。
    if not str(opt.store_name).endswith(str(opt.sample_duration)):
        opt.arch = 'model:{} fusion:{} num_heads:{}'.format(opt.model, opt.fusion, opt.num_heads)
        opt.store_name = '_'.join([opt.store_name, opt.arch, opt.dataset, str(opt.sample_duration)])

    snr_values = parse_snr_list(extra.snr_list)
    if extra.folds:
        fold_list = [int(x) for x in extra.folds.split(',') if x.strip() != '']
    else:
        fold_list = list(range(getattr(opt, 'n_folds', 1)))

    print('=' * 70)
    print('多梯度信噪比鲁棒性测试')
    print('=' * 70)
    print('融合方式   : {}'.format(opt.fusion))
    print('折号       : {}'.format(fold_list))
    print('信噪比档位 : {}'.format(['clean' if s is None else '{}dB'.format(s) for s in snr_values]))
    print('权重来源   : {}/<store_name>_best<折号>.pth'.format(opt.result_path))
    print('=' * 70)

    # 模型只构建一次，各折只是换权重。架构完全相同，没必要重复构建。
    model, _ = generate_model(opt)
    criterion = nn.CrossEntropyLoss().to(opt.device)

    os.makedirs(extra.dump_dir, exist_ok=True)

    # 关键：先把「未带折号」的原始标注路径存下来。
    # 循环里 opt.annotation_path 会被改写成本折文件，若直接在改写后的值上
    # 再拼折号，第 2 折起就会变成 ..._fold_0_1.txt 这种错误路径（q3 曾因此
    # 只跑出 fold 0 的数据）。
    orig_annotation_path = opt.annotation_path

    for fold in fold_list:
        # 折文件命名规则与 main.py 保持一致：<前缀去 .txt> + '_' + 折号 + '.txt'
        assert orig_annotation_path.endswith('.txt'), \
            'annotation_path 必须以 .txt 结尾，才能拼出各折文件'
        fold_annot = orig_annotation_path[:-4] + '_' + str(fold) + '.txt'
        if not os.path.isfile(fold_annot):
            print('[跳过] 第 {} 折：找不到 {}'.format(fold, fold_annot))
            continue

        ckpt_path = '{}/{}_best{}.pth'.format(opt.result_path, opt.store_name, fold)
        if not os.path.isfile(ckpt_path):
            print('[跳过] 第 {} 折：找不到权重 {}'.format(fold, ckpt_path))
            continue

        print('\n>>> 第 {} 折，加载 {}'.format(fold, ckpt_path))
        state = safe_torch_load(ckpt_path, map_location=opt.device)
        model.load_state_dict(state['state_dict'])

        # 注意：Logger 会断言「表头里的每一列都必须在 log 的字典里出现」，
        # 而 val_epoch 内部写死的字段就是下面这几个（与 main.py 的 test_logger 一致）。
        # 表头必须完全对齐，否则会抛 AssertionError。
        # 信噪比信息不放在这里，而是靠下面的标记行 + q3_snr.csv 记录。
        log_path = os.path.join(opt.result_path, 'robust_fold{}.log'.format(fold))
        logger = Logger(
            log_path,
            ['epoch', 'loss', 'loss1', 'loss2', 'loss3',
             'prec1_m', 'prec5_m', 'prec1_a', 'prec5_a', 'prec1_v', 'prec5_v',
             'bacc_m', 'bacc_a', 'bacc_v'])

        for snr in snr_values:
            tag = snr_tag(snr)
            opt.annotation_path = fold_annot
            opt.test_snr = snr  # None 表示干净；数值表示按该信噪比加噪

            test_data = get_test_set(opt, spatial_transform=transforms.Compose(
                [transforms.ToTensor(opt.video_norm_value)]))
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.n_threads, pin_memory=True,
                persistent_workers=opt.n_threads > 0)

            dump_path = os.path.join(
                extra.dump_dir, '{}_fold{}_snr{}.csv'.format(opt.store_name, fold, tag))

            print('    信噪比 {} ...'.format(tag))
            # 每档噪声前写一行标记，方便在日志里区分各档结果
            # （Logger 只在文件为空时写表头，所以这里直接写一行不会被当成数据）
            logger.logger.writerow(['------fold:', fold, '------snr_db:', tag, '------'])
            loss, prec_m, prec_a, prec_v, bacc_m, bacc_a, bacc_v = val_epoch(
                epoch=10000, data_loader=test_loader, model=model, criterion=criterion,
                opt=opt, logger=logger, modality='both', dist=None,
                dump_path=dump_path, snr_tag=tag, fold=fold)

            print('    信噪比 {:<6s} -> M={:.2f}%  A={:.2f}%  V={:.2f}%  (BA: M={:.2f} A={:.2f} V={:.2f})'.format(
                tag, prec_m, prec_a, prec_v, bacc_m, bacc_a, bacc_v))

            # 追加一行到问题3 的汇总表
            _append_snr_summary(extra.snr_summary_csv, fold, tag, loss, prec_m, prec_a, prec_v, bacc_m, bacc_a, bacc_v)

    print('\n全部完成。')
    print('  - 逐样本预测: {}/'.format(extra.dump_dir))
    print('  - 准确率汇总: {}'.format(extra.snr_summary_csv))
    print('  - 下一步: python analyze_stats.py 计算错分类型分布')
    print('             bash plot.sh 出图')


def _append_snr_summary(path, fold, snr_tag_value, loss, prec_m, prec_a, prec_v, bacc_m=0, bacc_a=0, bacc_v=0):
    """往信噪比汇总 CSV 追加一行。表头在文件为空时写入。"""
    import csv

    fields = ['fold', 'snr_db', 'loss', 'prec1_m', 'prec1_a', 'prec1_v', 'bacc_m', 'bacc_a', 'bacc_v']
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({
            'fold': fold,
            'snr_db': snr_tag_value,
            'loss': round(float(loss), 4),
            'prec1_m': round(float(prec_m), 4),
            'prec1_a': round(float(prec_a), 4),
            'prec1_v': round(float(prec_v), 4),
            'bacc_m': round(float(bacc_m), 4),
            'bacc_a': round(float(bacc_a), 4),
            'bacc_v': round(float(bacc_v), 4),
        })


if __name__ == '__main__':
    main()
