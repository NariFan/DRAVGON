# -*- coding: utf-8 -*-
"""
数据集构造入口。main.py 通过本文件的三个函数拿到训练/验证/测试集。

本文件最近一次修改是为回应审稿人意见（补充 K 折交叉验证与噪声鲁棒性实验）。
改动原则：
  1. 三个函数的**对外签名完全不变**，main.py 的调用方式无需修改；
  2. 新增能力全部通过 opt 的可选字段读取，读不到就退回原有行为，
     因此老的命令行参数组合仍然能原样跑通。
"""

from datasets.ravdess import RAVDESS


def _dataset_kwargs(opt):
    """
    从 opt 里抽取「公共的数据集构造参数」。

    之所以用 getattr 而不是直接 opt.xxx：这些参数都是后加的，
    如果某段老代码传入的 opt 没有这些字段（比如自己拼的命名空间），
    也不会抛 AttributeError，而是安静地退回默认行为。

    BEATs 权重路径、噪声随机种子等都属于这一类。
    """
    return dict(
        beats_ckpt=getattr(opt, 'beats_ckpt', './pre-trained_checkpoints/BEATs_iter3.pt'),
        snr_power_mode=getattr(opt, 'snr_power_mode', 'active'),
        snr_top_db=getattr(opt, 'snr_top_db', 20),
        noise_seed=getattr(opt, 'noise_seed', 1234),
        beats_cache_dir=getattr(opt, 'beats_cache_dir', ''),
        audio_backbone=getattr(opt, 'audio_backbone', 'beats'),
        emotion2vec_dir=getattr(opt, 'emotion2vec_dir', './pre-trained_checkpoints/emotion2vec'),
        # HAR-Learner 消融：控制声学特征里是否含 MFCC / CQT / 深度特征
        har=getattr(opt, 'har', 'full'),
    )


# RAVDESS 与 PKU-ER 的标注格式完全相同（video;audio;label;subset），共用一套 loader。
# 以前这里 assert 只允许 'RAVDESS'，PKU-ER 实验被迫顶着 RAVDESS 的名字跑，
# 于是磁盘上出现了 xxx_RAVDESS_30_best2.pth 这种自相矛盾的文件名。
# 现在两个名字都放行：opt.dataset 只决定命名，加载逻辑不变。
_DATASETS = ['RAVDESS', 'PKU-ER', 'PKU_ER', 'PKU']


def _check_dataset(opt):
    if opt.dataset not in _DATASETS:
        raise ValueError('Unsupported dataset: {}（支持 {}）'.format(opt.dataset, _DATASETS))


def get_training_set(opt, spatial_transform=None, audio_transform=None):
    _check_dataset(opt)

    if True:
        training_data = RAVDESS(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisual', audio_transform=audio_transform,
            # 训练集永远不加信噪比噪声。
            # 噪声是「测试时模拟真实环境干扰」的手段（审稿人问题3），
            # 训练侧已经有 --mask 做模态丢弃增强，两者不应混用。
            snr=None, **_dataset_kwargs(opt))
    return training_data


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    _check_dataset(opt)

    if True:
        validation_data = RAVDESS(
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform, data_type = 'audiovisual', audio_transform=audio_transform,
            # 验证集保持干净：它负责挑选 best checkpoint，
            # 若验证集也被污染，选出的模型会偏向抗噪而牺牲干净场景的精度。
            snr=None, **_dataset_kwargs(opt))
    return validation_data


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    _check_dataset(opt)
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if True:
        test_data = RAVDESS(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform,
            # 测试集才施加信噪比噪声。opt.test_snr 为 None 即干净测试集，行为与修改前一致。
            snr=getattr(opt, 'test_snr', None), **_dataset_kwargs(opt))
    return test_data
