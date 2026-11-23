# -*- coding: utf-8 -*-
"""
emotion2vec+ 音频特征提取模块（BEATs 的可替换基座）。

动机
----
审稿人 C1 指出 BEATs 是 AudioSet 通用音频事件预训练，与情感目标域存在
domain bias。emotion2vec+（复旦 Speech Lab, 2024）是**情感语音专用**的
SSL 预训练模型，与 AVER 任务的 domain 完全匹配，作为替换基座的对照实验。

设计原则
--------
1. 与 BEATs 输出形状完全一致：(768, T) -> 取前 156 帧 -> (768, 156)。
   emotion2vec_plus_base 的 embed_dim=768，与 BEATs 同维，
   因此 datasets/ravdess.py 里的 820 通道拼接（MFCC 10 + CQT 42 + 基座 768）
   和 models/multimodalcnn.py 的切片常量**完全不用改**。
2. 加载用 FunASR 官方 AutoModel（本目录下 model.pt + config.yaml）。
3. 帧率：data2vec audio 的卷积步长 320 @16kHz = 20ms → 3.6s 约 180 帧，
   取前 156 帧即可（与 BEATs 截断逻辑一致）。

权重获取：hf-mirror.com/emotion2vec/emotion2vec_plus_base
"""

import os

import torch

_E2V_FRAME_TARGET = 156  # 与 BEATs 截断帧数一致


def load_emotion2vec_model(ckpt_dir, device='cpu'):
    """
    加载 emotion2vec+ 模型。返回一个可调用的特征提取器（eval 模式）。
    ckpt_dir 内需有 model.pt 和 config.yaml。
    """
    from funasr import AutoModel
    m = AutoModel(
        model=ckpt_dir,
        disable_update=True,
        device=device,
    )
    m.model.eval()
    return m


def extract_emotion2vec_features(model, y_16k, device='cpu'):
    """
    输入：16kHz 波形（numpy 或 1D tensor）。
    输出：(768, 156) float32 tensor，与 get_beats 完全同形。

    实现对齐 funasr.models.emotion2vec.model.Emotion2vec.inference 的内部流程：
      1. 波形做 layer_norm（cfg.normalize=True 时的标准预处理，不能漏，
         否则特征分布与官方推理不一致）；
      2. extract_features(source, padding_mask=None) -> feats['x'] 为 (1, T, 768)。
    """
    import torch.nn.functional as F

    if torch.is_tensor(y_16k):
        y = y_16k.float()
    else:
        y = torch.from_numpy(y_16k).float()
    y = y.view(1, -1).to(device)

    with torch.no_grad():
        # 官方推理代码中的 normalize 步骤（config.yaml 里 normalize: true）
        if getattr(model.model.cfg, 'normalize', False):
            y = F.layer_norm(y, y.shape)
        out = model.model.extract_features(y, padding_mask=None)
        feats = out['x']  # (1, T, 768)

    feats = feats.squeeze(0)          # (T, 768)
    feats = feats[:_E2V_FRAME_TARGET, :].transpose(0, 1).cpu().float()  # (768, 156)
    return feats


def extract_emotion2vec_from_wav_22050(model, y_22050, device='cpu'):
    """
    与 datasets/ravdess.py 的 get_beats 完全同构的入口：
    输入 22050Hz 波形，内部重采样到 16kHz，输出 (768, 156)。
    """
    import librosa
    y16 = librosa.resample(y_22050, orig_sr=22050, target_sr=16000)
    return extract_emotion2vec_features(model, y16, device=device)
