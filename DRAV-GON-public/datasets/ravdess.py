# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import os

import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa

# 波形层信噪比加噪（回应审稿人问题3）。见 noise_utils.py 顶部说明。
from noise_utils import add_noise_at_snr
from utils import safe_torch_load



def video_loader(video_dir_path):
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr=sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc = torch.tensor(mfcc,dtype=torch.float32)
    return mfcc

def get_l0(y, sr):
    l0 = np.abs(y)
    return l0

def get_cqt(y,sr):
    cqt = librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=42, bins_per_octave=12)
    cqt = torch.from_numpy(cqt.real).to(torch.float32)
    return cqt



def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, audiofilename, label, trainvaltest = line.split(';')        
        if trainvaltest.rstrip() != subset:
            continue
        
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None,
                 # ↓↓↓ 以下为审稿人补充实验新增的参数，全部留 None/默认值即保持原有行为 ↓↓↓
                 beats_ckpt='./pre-trained_checkpoints/BEATs_iter3.pt',
                 snr=None, snr_power_mode='active', snr_top_db=20, noise_seed=1234,
                 beats_cache_dir='',
                 audio_backbone='beats',
                 emotion2vec_dir='./pre-trained_checkpoints/emotion2vec',
                 har='full'):
        # HAR-Learner 消融开关。full=MFCC+CQT+深度特征；deep=只留深度特征；
        # handcrafted=只留 MFCC+CQT。被关掉的那几路直接填 0，不再跑 librosa。
        # 通道布局恒为 820（10+42+768），这样模型侧的切片常量一个都不用改。
        assert har in ('full', 'deep', 'handcrafted'), 'har 必须是 full | deep | handcrafted'
        self.har = har
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type

        # BEATs 权重路径。原来是硬编码的，现改为可配（默认值与原来一致）。
        # 注意这是相对路径，相对的是「运行 main.py 时的工作目录」，
        # 所以所有命令都必须在项目根目录 DRAV-GON/ 下执行。
        self.beats_ckpt = beats_ckpt

        # 音频基座选择：'beats'（默认，原行为）或 'emotion2vec'（审稿人 C1 对照实验）。
        # 两者输出形状完全一致 (768,156)，820 通道拼接与模型切片常量均无需改动。
        self.audio_backbone = audio_backbone
        self.emotion2vec_dir = emotion2vec_dir

        # 信噪比加噪配置。snr=None 表示不加噪（干净样本）。
        # snr 为数值时，在波形层叠加对应信噪比(dB)的高斯白噪声。
        self.snr = snr
        self.snr_power_mode = snr_power_mode
        self.snr_top_db = snr_top_db
        self.noise_seed = noise_seed

        # BEATs 特征缓存目录。留空表示不用缓存（与改动前行为一致）。
        # 只在 snr is None（不加噪）时才会去读缓存 —— 加噪场景下噪声必须真的
        # 流入 BEATs，用缓存会让问题3 的鲁棒性实验失去意义。
        # 注意：BEATs 与 emotion2vec 的缓存必须放在不同目录（由调用方控制），
        # 同名文件内容不同，混用会静默拿到错误的特征。
        self.beats_cache_dir = beats_cache_dir

        # 在初始化时加载模型。
        # 例外：明确要用缓存且不加噪时（训练/干净测试），先不加载 ——
        # DataLoader 每个 worker 都会构造一次 dataset，不加载能省下
        # 每个 worker 一份 345MB 的 BEATs 常驻内存和几十秒的加载时间。
        # 万一某条音频缓存缺失，get_beats 会懒加载兜底。
        if beats_cache_dir and snr is None:
            self.beats_model = None
        else:
            self.beats_model = self.load_beats_model()

    def load_beats_model(self):
        if self.audio_backbone == 'emotion2vec':
            from emotion2vec_feat import load_emotion2vec_model
            return load_emotion2vec_model(self.emotion2vec_dir, device='cpu')
        from beats.BEATs import BEATs, BEATsConfig
        checkpoint = safe_torch_load(self.beats_ckpt)
        cfg = BEATsConfig(checkpoint['cfg'])
        beats_model = BEATs(cfg)
        beats_model.load_state_dict(checkpoint['model'])
        beats_model.eval()
        return beats_model

    def get_beats(self, y, sr):
        if self.beats_model is None:  # 缓存缺失时的懒加载兜底
            self.beats_model = self.load_beats_model()
        if sr != 16000:  # 假设模型期望的采样率为16kHz
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # emotion2vec 分支：输出与 BEATs 完全同形 (768,156)，见 emotion2vec_feat.py
        if self.audio_backbone == 'emotion2vec':
            from emotion2vec_feat import extract_emotion2vec_features
            return extract_emotion2vec_features(self.beats_model, y, device='cpu')

        y = torch.from_numpy(y).float()  # 转换为 float 类型的张量
        audio_input = y.unsqueeze(0)  # 添加批次维度
        padding_mask = torch.zeros(audio_input.shape, dtype=torch.bool)

        with torch.no_grad():
            representation = self.beats_model.extract_features(audio_input, padding_mask=padding_mask)[0]

        # 直接对 Tensor 进行操作
        representation = representation.squeeze()  # 去掉维度为1的维度
        representation = representation[:156, :]  # 选择前156行
        representation = representation.transpose(0, 1)  # 转置

        return representation

    def _load_cached_audio_features(self, audio_path):
        """
        读「全特征缓存」（MFCC + CQT + BEATs 一个文件全有，见 extract_beats_cache.py）。

        返回 (mfcc, cqt, beats)；不能用缓存时返回 None，调用方走在线计算：
          - 未启用缓存 / 文件缺失
          - 加噪样本（snr is None 才允许缓存，理由同 _get_beats_or_cache）
          - 旧版缓存文件里只有 BEATs 张量（dict 才是全特征格式）

        命中时连 wav 都不需要读，直接省掉 librosa 全套计算 ——
        这是训练速度的主要瓶颈（6 核 CPU 上 CQT 一条就要上百毫秒）。
        """
        if not (self.beats_cache_dir and self.snr is None):
            return None
        from extract_beats_cache import cache_path_for
        f = cache_path_for(audio_path, self.beats_cache_dir)
        if not os.path.isfile(f):
            return None
        d = safe_torch_load(f)
        if isinstance(d, dict) and 'mfcc' in d and 'cqt' in d and 'beats' in d:
            return d['mfcc'], d['cqt'], d['beats']
        return None

    def _get_beats_or_cache(self, audio_path, y, sr):
        """
        取 BEATs 特征：能读缓存就读，读不到就退回在线计算。

        缓存只在「不加噪」时启用。加噪样本（审稿人问题3）必须在线重算，
        否则噪声进不了 BEATs，鲁棒性结论就站不住脚。
        """
        if self.snr is None and self.beats_cache_dir:
            from extract_beats_cache import cache_path_for
            cache_file = cache_path_for(audio_path, self.beats_cache_dir)
            if os.path.isfile(cache_file):
                return safe_torch_load(cache_file)
        return self.get_beats(y, sr)

    def _apply_har(self, mfcc, cqt, beats):
        """
        按 har 决定 820 通道里哪几路真的带信息。布局恒为 (820, T)。
        模型侧 AudioCNNPool 只实例化启用的塔，两边必须一致。
        """
        if self.har == 'deep':
            mfcc = torch.zeros_like(mfcc)
            cqt = torch.zeros_like(cqt)
        elif self.har == 'handcrafted':
            beats = torch.zeros_like(beats)
        return torch.cat((mfcc, cqt, beats), dim=0)

    def __getitem__(self, index):
        target = self.data[index]['label']
                

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']

            # 优先读全特征缓存（不加噪时）。命中则跳过 wav 读取和 librosa 全套计算。
            cached = self._load_cached_audio_features(path)
            if cached is not None:
                mfcc, cqt, beats = cached
            elif self.har == 'deep':
                # 只留深度特征：MFCC / CQT 完全不计算，省掉两次 librosa
                y, sr = load_audio(path, sr=22050)
                if self.audio_transform is not None:
                    self.audio_transform.randomize_parameters()
                    y = self.audio_transform(y)
                if self.snr is not None:
                    y, _ = add_noise_at_snr(
                        y, self.snr,
                        sample_key=os.path.basename(path),
                        seed=self.noise_seed,
                        mode=self.snr_power_mode,
                        top_db=self.snr_top_db)
                beats = self._get_beats_or_cache(path, y, sr)
                mfcc = torch.zeros(10, beats.shape[1], dtype=torch.float32)
                cqt = torch.zeros(42, beats.shape[1], dtype=torch.float32)
            elif self.har == 'handcrafted':
                # 只留手工频谱特征：跳过整条深度特征通路（BEATs / emotion2vec 前向）
                y, sr = load_audio(path, sr=22050)
                if self.audio_transform is not None:
                    self.audio_transform.randomize_parameters()
                    y = self.audio_transform(y)
                if self.snr is not None:
                    y, _ = add_noise_at_snr(
                        y, self.snr,
                        sample_key=os.path.basename(path),
                        seed=self.noise_seed,
                        mode=self.snr_power_mode,
                        top_db=self.snr_top_db)
                mfcc = get_mfccs(y, sr)
                cqt = get_cqt(y, sr)
                beats = torch.zeros(768, mfcc.shape[1], dtype=torch.float32)
            else:
                y, sr = load_audio(path, sr=22050)

                if self.audio_transform is not None:
                     self.audio_transform.randomize_parameters()
                     y = self.audio_transform(y)

                # === 波形层信噪比加噪（回应审稿人问题3）===
                # 位置很关键：必须在提取 MFCC / CQT / BEATs 特征「之前」，
                # 这样三种声学特征会一起被污染，模拟真实环境下的录音噪声，
                # 而不是只对特征做扰动（后者在审稿人那里站不住脚）。
                #
                # self.snr 为 None 表示干净测试集，直接跳过；
                # 用音频文件名（basename）而非样本下标播种，
                # 这样不受标注行序、DataLoader 多进程顺序影响，同一条样本在
                # 不同折上拿到的是同一段噪声，跨折结果可比、论文可复现。
                if self.snr is not None:
                    y, _ = add_noise_at_snr(
                        y, self.snr,
                        sample_key=os.path.basename(path),
                        seed=self.noise_seed,
                        mode=self.snr_power_mode,
                        top_db=self.snr_top_db)

                mfcc = get_mfccs(y, sr)
                l0 = get_l0(y,sr)
                cqt = get_cqt(y,sr)
                beats = self._get_beats_or_cache(path, y, sr)
            # print('mfcc:',mfcc.shape) #mfcc: (10, 156)
            # print('l0:',l0[:, np.newaxis].shape) #l0: (79380, 1)
            # print('cqt:',cqt.shape) #cqt (84, 156)
            # print('beats:',beats.shape)
            # 通道布局恒为 820；被 har 关掉的段再兜一次底归零（缓存路径会带上完整三路）
            concatenated_features = self._apply_har(mfcc, cqt, beats)
            audio_features = concatenated_features

            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  

    def __len__(self):
        return len(self.data)
