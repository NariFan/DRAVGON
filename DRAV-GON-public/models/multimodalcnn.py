# -*- coding: utf-8 -*-
"""
Parts of this code are based on https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py
"""

import torch
import torch.nn as nn
from models.modulator import Modulator
from models.efficientface import LocalFeatureExtractor, InvertedResidual
from models.transformer_timm import AttentionBlock, Attention
from utils import safe_torch_load

def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True)) 

class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=7, im_per_sample=25):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)
        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )
        self.im_per_sample = im_per_sample
        
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) #global average pooling
        return x

    def forward_stage1(self, x):
        #Getting samples per batch
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0,2,1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
        
        
    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x
    
    def forward_classifier(self, x):
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x
        
      

def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = safe_torch_load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficientnet')
    model.load_state_dict(pre_trained_dict, strict=False)

    
def get_model(num_classes, task, seq_length):
    model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, task, seq_length)
    return model  


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding='valid'),nn.BatchNorm1d(out_channels),
                                   nn.ReLU(inplace=True), nn.MaxPool1d(2,1))

class AudioCNNPool(nn.Module):
    """
    声学分支。三个子塔（MFCC / CQT / 深度特征）各自卷到 128 维后相加。

    har 控制「异构」程度，即 HAR-Learner 的消融开关：
        full         -> 三塔全建（MFCC + CQT + 深度特征），完整 HAR-Learner
        deep         -> 只建深度特征塔（w/o HAR-Learner）
        handcrafted  -> 只建 MFCC + CQT 两塔（反向对照）
    未启用的塔干脆不实例化，这样论文里报的参数量才是真实的。
    """

    def __init__(self, num_classes=8, har='full'):
        super(AudioCNNPool, self).__init__()

        input_channels_mfcc = 10
        input_channels_cqt = 42
        input_channels_beats = 768

        self.har = har
        self.use_mfcc = har in ('full', 'handcrafted')
        self.use_cqt = har in ('full', 'handcrafted')
        self.use_deep = har in ('full', 'deep')
        assert any([self.use_mfcc, self.use_cqt, self.use_deep]), 'har 至少启用一路'

        if self.use_mfcc:
            self.conv1d_0 = conv1d_block_audio(input_channels_mfcc, 64)
            self.conv1d_1 = conv1d_block_audio(64, 128)
        if self.use_cqt:
            self.conv1d_4 = conv1d_block_audio(input_channels_cqt, 64)
            self.conv1d_5 = conv1d_block_audio(64, 128)
        if self.use_deep:
            self.conv1d_6 = conv1d_block_audio(input_channels_beats, 64)
            self.conv1d_7 = conv1d_block_audio(64, 128)

        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.classifier_1 = nn.Sequential(
                nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage3(self, x):
        x = self.conv1d_4(x)
        x = self.conv1d_5(x)
        return x

    def forward_stage4(self, x):
        x = self.conv1d_6(x)
        x = self.conv1d_7(x)
        return x

    def forward_hetero(self, x_audio):
        """
        按 har 把 820 通道切成三段、只跑启用的塔、相加。
        恒返回 (B, 128, T)，形状与原实现一致。

        注意：这里是唯一合法的声学入口。直接调 forward_stage1/3/4 会在
        har != full 时因为子塔不存在而崩，各 forward_feature_* 都已改用本函数。
        """
        parts = []
        if self.use_mfcc:
            parts.append(self.forward_stage1(x_audio[:, 0:10, :]))
        if self.use_cqt:
            parts.append(self.forward_stage3(x_audio[:, 10:52, :]))
        if self.use_deep:
            parts.append(self.forward_stage4(x_audio[:, 52:820, :]))
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def forward_stage1(self,x):            
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x
    
    def forward_stage2(self,x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)   
        return x
    
    def forward_classifier(self, x):   
        x = x.mean([-1]) #pooling accross temporal dimension
        x1 = self.classifier_1(x)
        return x1

    


class MultiModalCNN(nn.Module):
    """
    消融三轴全部体现在构造参数里，并会出现在自动生成的实验名中：
        fusion      : concat | lt | it | ia | gs_it | gs_lt | gs_ia
        har         : full | deep | handcrafted   (HAR-Learner)
        cross_dir   : both | av | va              (双向 / 单向)
        cross_depth : 单向时同向堆叠层数（容量匹配）
    """

    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef='None', num_heads=1,
                 har='full', cross_dir='both', cross_depth=1):
        super(MultiModalCNN, self).__init__()
        assert fusion in ['concat', 'ia', 'it', 'lt', 'gs_it', 'gs_ia', 'gs_lt'], \
            print('Unsupported fusion method: {}'.format(fusion))
        assert cross_dir in ['both', 'av', 'va'], print('Unsupported cross_dir: {}'.format(cross_dir))
        assert har in ['full', 'deep', 'handcrafted'], print('Unsupported har: {}'.format(har))

        self.audio_model = AudioCNNPool(num_classes=num_classes, har=har)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)

        init_feature_extractor(self.visual_model, pretr_ef)

        e_dim = 128
        input_dim_video = 128
        input_dim_audio = 128
        self.fusion = fusion
        self.har = har
        self.cross_dir = cross_dir
        # both 时深度恒为 1（双向本身就是两个方向各一层），填其他值没有意义。
        self.cross_depth = 1 if cross_dir == 'both' else max(1, int(cross_depth))
        self.mfcc_start, self.mfcc_end = 0, 10
        self.cqt_start, self.cqt_end = 10, 52
        self.beats_start, self.beats_end = 52, 820

        # 每种 fusion 的 av / va 形状配置（与原实现逐字对应，只是抽成 dict 便于复用）
        if fusion == 'lt':                                            # forward_transformer
            av_cfg = dict(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim)
            va_cfg = dict(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim)
            input_dim_video = input_dim_video // 2
            block_cls = AttentionBlock
        elif fusion == 'gs_lt':                                       # forward_feature_5
            av_cfg = dict(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio)
            va_cfg = dict(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video)
            input_dim_video = input_dim_video // 2
            block_cls = AttentionBlock
        elif fusion == 'it':                                          # forward_feature_3
            input_dim_video = input_dim_video // 2
            av_cfg = dict(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio)
            va_cfg = dict(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video)
            block_cls = AttentionBlock
        elif fusion == 'gs_it':                                       # forward_feature_1
            input_dim_video = input_dim_video // 2
            av_cfg = dict(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio)
            va_cfg = dict(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video)
            block_cls = AttentionBlock
        elif fusion == 'ia':                                          # forward_feature_2
            input_dim_video = input_dim_video // 2
            av_cfg = dict(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio)
            va_cfg = dict(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video)
            block_cls = Attention
        elif fusion == 'gs_ia':                                       # forward_feature_4
            input_dim_video = input_dim_video // 2
            av_cfg = dict(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim)
            va_cfg = dict(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim)
            block_cls = Attention
        else:                                                         # concat
            # 无交叉注意力、无 Gumbel 采样融合：两路各自池化后直接拼接。
            input_dim_video = input_dim_video // 2
            av_cfg = va_cfg = None
            block_cls = None

        if block_cls is None:
            self.av_blocks = nn.ModuleList()
            self.va_blocks = nn.ModuleList()
        else:
            self._build_cross_blocks(av_cfg, va_cfg, num_heads, block_cls)

        self.fusion_linear_sem = nn.Linear(e_dim, e_dim)  # Assuming input dim=128
        self.fusion_linear_spe = nn.Linear(e_dim, e_dim)
        self.classifier_1 = nn.Sequential(
                    nn.Linear(e_dim*2, num_classes),
                )
        self.classifier_2 = nn.Sequential(
            nn.Linear(input_dim_audio, num_classes),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(input_dim_video, num_classes),
        )

    # ------------------------------------------------------------------
    # 交叉注意力：方向（cross_dir）与深度（cross_depth）
    # ------------------------------------------------------------------
    def _build_cross_blocks(self, av_cfg, va_cfg, num_heads, block_cls):
        """
        按 cross_dir 建 av / va 两组 block。

        av = 音频查询视觉（用视觉去增强音频）；va = 视觉查询音频。

            cross_dir='both' -> av 一层 + va 一层（论文的结构性对称交互）
            cross_dir='av'   -> 只有 av，堆 cross_depth 层
            cross_dir='va'   -> 只有 va，堆 cross_depth 层

        单向时每层形状完全相同，所以总参数量 = cross_depth × 单层参数量，
        这就是「容量匹配」的旋钮：
            gs_it 实测 av 单层 115.8k、va 单层 41.7k、双向合计 157.5k
            -> av×2 = 231.6k、va×4 = 166.8k，均可让单向反超双向的参数量。
        """
        if self.cross_dir == 'both':
            n_av, n_va = 1, 1
        elif self.cross_dir == 'av':
            n_av, n_va = self.cross_depth, 0
        else:
            n_av, n_va = 0, self.cross_depth

        # 堆叠（cross_depth>1）要求该方向的 out_dim 等于自己的 in_dim_q，
        # 否则第二层输入的维度和它声明的 in_dim_q 对不上。
        # 本项目的消融只用 gs_it（av:128->128、va:64->64，两条都满足）；
        # gs_ia 的 va 是 64->128，不能堆叠，这里直接报错而不是等到 forward 才炸。
        for tag, n, cfg in (('av', n_av, av_cfg), ('va', n_va, va_cfg)):
            if n > 1 and cfg['out_dim'] != cfg['in_dim_q']:
                raise ValueError(
                    'cross_dir=%s 无法堆叠 %d 层：out_dim=%d != in_dim_q=%d'
                    % (tag, n, cfg['out_dim'], cfg['in_dim_q']))

        self.av_blocks = nn.ModuleList(
            [block_cls(num_heads=num_heads, **av_cfg) for _ in range(n_av)])
        self.va_blocks = nn.ModuleList(
            [block_cls(num_heads=num_heads, **va_cfg) for _ in range(n_va)])

    def _run_av(self, xk, xq):
        """音频查询视觉。该方向被关掉时返回 None，调用方据此跳过残差相加。"""
        if len(self.av_blocks) == 0:
            return None
        for blk in self.av_blocks:
            xq = blk(xk, xq)
        return xq

    def _run_va(self, xk, xq):
        """视觉查询音频。该方向被关掉时返回 None，调用方据此跳过残差相加。"""
        if len(self.va_blocks) == 0:
            return None
        for blk in self.va_blocks:
            xq = blk(xk, xq)
        return xq

    def _attn_matrix(self, blocks, xk, xq):
        """
        ia / gs_ia 用的是 Attention 返回的注意力矩阵当门控（不是块输出）。
        该方向被关掉时返回 None。
        """
        if len(blocks) == 0:
            return None
        attn = None
        for blk in blocks:
            xq, attn = blk(xk, xq)
        return attn

    def cross_params(self):
        """交叉注意力的参数量，写论文 Params 列时用。"""
        return sum(p.numel() for b in self.av_blocks for p in b.parameters()) \
             + sum(p.numel() for b in self.va_blocks for p in b.parameters())

    def gumbel_sigmoid(self, logits, tau=1.0, hard=False):
            # 添加Gumbel噪声
            noise = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
            tau = torch.tensor(tau, device=logits.device)
            tau = torch.clamp(tau, min=0.1, max=1.0)
            y = (logits + gumbel_noise) / tau
            return torch.sigmoid(y)       
    #def gumbel_sigmoid(self, logits, tau=1.0, hard=True):
        #logits = torch.stack([logits, -logits], dim=-1)
        #gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
        #samples = torch.softmax((logits + gumbels)/tau, dim=-1)
        #if hard:
            #samples_hard = torch.zeros_like(samples).scatter_(-1, samples.argmax(dim=-1, keepdim=True), 1.0)
            #samples = samples_hard - samples.detach() + samples
        #return samples[..., 0]
        
    def forward(self, x_audio, x_visual, tau=1):
        if self.fusion == 'concat':
            return self.forward_concat(x_audio, x_visual)
        elif self.fusion == 'lt':
            return self.forward_transformer(x_audio, x_visual)
        elif self.fusion == 'ia':
            return self.forward_feature_2(x_audio, x_visual)
        elif self.fusion == 'it':
            return self.forward_feature_3(x_audio, x_visual)
        elif self.fusion == 'gs_it':
            return self.forward_feature_1(x_audio, x_visual, tau)
        elif self.fusion == 'gs_ia':
            return self.forward_feature_4(x_audio, x_visual, tau)
        elif self.fusion == 'gs_lt':
            return self.forward_feature_5(x_audio, x_visual, tau)

    def forward_concat(self, x_audio, x_visual):
        """
        消融下界：完全去掉 G²AF —— 既无交叉注意力、也无 Gumbel 采样融合，
        两路各自池化后直接拼接送分类器。三个预测头照旧保留。
        """
        return self._two_stream(x_audio, x_visual, gumbel=False)

    def _two_stream(self, x_audio, x_visual, gumbel=False, tau=0.1):
        """
        it / gs_it / concat 共用的主干。

        gumbel=False 时最后一步是朴素拼接（对应 --fusion it 与 concat 的融合部分）；
        gumbel=True  时走 Gumbel 采样融合（对应 --fusion gs_it）。
        交叉注意力是否生效由 self.av_blocks / self.va_blocks 是否为空决定，
        concat 下两个列表都是空的，于是自然退化成「无交互」。
        """
        x_audio = self.audio_model.forward_hetero(x_audio)

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        # 辅助单模态头用的是「融合前」的池化特征
        x_visual_ = x_visual.permute(0, 2, 1).mean([-2])
        x_audio_ = x_audio.permute(0, 2, 1).mean([-2])

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        # 交叉注意力：av 用视觉增强音频，va 用音频增强视觉。
        # 某一方向被关掉（单向消融 / concat）时 None，该支路就完全不更新。
        h_av = self._run_av(proj_x_v, proj_x_a)
        h_va = self._run_va(proj_x_a, proj_x_v)
        if h_av is not None:
            x_audio = h_av.permute(0, 2, 1) + x_audio
        if h_va is not None:
            x_visual = h_va.permute(0, 2, 1) + x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])
        video_pooled = x_visual.mean([-1])

        if gumbel:
            x = self._gumbel_fusion(audio_pooled, video_pooled, tau)
        else:
            x = torch.cat((audio_pooled, video_pooled), dim=-1)

        return {
            'M': self.classifier_1(x),
            'A': self.classifier_2(x_audio_),
            'V': self.classifier_3(x_visual_),
        }

    def _gumbel_fusion(self, f_sem, f_spe, tau):
        """G²AF 的第二阶段：特征交互 -> Gumbel 采样 -> 残差融合 -> 拼接。"""
        f_sem_transformed = torch.relu(self.fusion_linear_sem(f_spe))
        f_spe_transformed = torch.relu(self.fusion_linear_spe(f_sem))

        f_sem_prime = f_sem * f_sem_transformed
        f_spe_prime = f_spe * f_spe_transformed

        s_sem = self.gumbel_sigmoid(f_sem, tau=tau, hard=False)
        s_spe = self.gumbel_sigmoid(f_spe, tau=tau, hard=False)

        h_sem = f_sem_prime * s_sem + f_sem
        h_spe = f_spe_prime * s_spe + f_spe

        return torch.cat([h_spe, h_sem], dim=-1)

 
        
    def forward_feature_3(self, x_audio, x_visual):
        # --fusion it：交叉注意力 + 朴素拼接（去掉 Gumbel 采样融合）
        return self._two_stream(x_audio, x_visual, gumbel=False)

    def forward_feature_2(self, x_audio, x_visual):
        # 使用切片操作分别取出特征
        x_audio = self.audio_model.forward_hetero(x_audio)

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual) #(batchsize,64,15)

        x_visual_ = x_visual.permute(0,2,1).mean([-2]) #(batchsize,64)
        x_audio_ = x_audio.permute(0,2,1).mean([-2]) #(batchsize,128)

        proj_x_a = x_audio.permute(0,2,1)   #(batchsize,150,128)
        proj_x_v = x_visual.permute(0,2,1)  #(batchsize,15,64)

        h_av = self._attn_matrix(self.av_blocks, proj_x_v, proj_x_a) # (batchsize,head,150,15)
        h_va = self._attn_matrix(self.va_blocks, proj_x_a, proj_x_v) # (batchsize,head,15,150)
        
        # 单向时 h_av / h_va 可能是 None（该方向被关掉）
        if h_av is not None:
            if h_av.size(1) > 1: #if more than 1 head, take average
                h_av = torch.mean(h_av, axis=1).unsqueeze(1)
            h_av = h_av.sum([-2])
        if h_va is not None:
            if h_va.size(1) > 1:
                h_va = torch.mean(h_va, axis=1).unsqueeze(1)
            h_va = h_va.sum([-2])

        if h_va is not None:
            x_audio = h_va*x_audio
        if h_av is not None:
            x_visual = h_av*x_visual
        
        x_audio = self.audio_model.forward_stage2(x_audio)       
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
        video_pooled = x_visual.mean([-1])
        
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        

        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x_audio_)
        x3 = self.classifier_3(x_visual_)
        ret = {
            'M': x1,
            'A': x2,
            'V': x3,
        }
        return ret
        # return x1

    def forward_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_hetero(x_audio)
        # x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)
       
        x_visual = self.visual_model.forward_features(x_visual) 
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        x_visual_ = x_visual.permute(0,2,1).mean([-2])
        x_audio_ = x_audio.permute(0,2,1).mean([-2])
           
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self._run_av(proj_x_v, proj_x_a)
        h_va = self._run_va(proj_x_a, proj_x_v)
        # 单向时该方向没有输出，退回未被注意的投影本身，形状不变
        audio_pooled = (proj_x_a if h_av is None else h_av).mean([1])
        video_pooled = (proj_x_v if h_va is None else h_va).mean([1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)  
        # x1 = self.classifier_1(x)
        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x_audio_)
        x3 = self.classifier_3(x_visual_)
        ret = {
            'M': x1,
            'A': x2,
            'V': x3,
        }
        return ret
    def forward_feature_5(self, x_audio, x_visual,tau=0.1):
        x_audio = self.audio_model.forward_hetero(x_audio)
        # x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)
       
        x_visual = self.visual_model.forward_features(x_visual) 
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        x_visual_ = x_visual.permute(0,2,1).mean([-2])
        x_audio_ = x_audio.permute(0,2,1).mean([-2])
           
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self._run_av(proj_x_v, proj_x_a)
        h_va = self._run_va(proj_x_a, proj_x_v)
        # 单向时该方向没有输出，退回未被注意的投影本身，形状不变
        audio_pooled = (proj_x_a if h_av is None else h_av).mean([1])
        video_pooled = (proj_x_v if h_va is None else h_va).mean([1])

        #x = torch.cat((audio_pooled, video_pooled), dim=-1)
        # =============================================
        # 5. Replace concat with Sampling Fusion
        # =============================================
        # Step 1: Feature Transformation
        f_sem = audio_pooled  # Treat audio as "semantic" feature
        f_spe = video_pooled  # Treat visual as "spectral" feature

        #维度匹配的特征变换 Linear layers to double dimensions (assuming input dim=128, output dim=128)
        f_sem_transformed = torch.relu(self.fusion_linear_sem(f_spe))  # Linear(f_spe)
        f_spe_transformed = torch.relu(self.fusion_linear_spe(f_sem))  # Linear(f_sem)

        # Element-wise multiplication
        f_sem_prime = f_sem * f_sem_transformed
        f_spe_prime = f_spe * f_spe_transformed

        # Step 2: Gumbel-Softmax Sampling
        s_sem = self.gumbel_sigmoid(f_sem, tau=tau, hard=False)
        s_spe = self.gumbel_sigmoid(f_spe, tau=tau, hard=False)

        # Step 3: Final Fusion 残差连接提升稳定性
        h_sem = f_sem_prime * s_sem + f_sem
        h_spe = f_spe_prime * s_spe + f_spe

        # Concatenate the fused features (now more robust than naive concat)
        x = torch.cat([h_spe, h_sem], dim=-1)
        # ============================================= 
        # x1 = self.classifier_1(x)
        x1 = self.classifier_1(x)
        x2 = self.classifier_2(x_audio_)
        x3 = self.classifier_3(x_visual_)
        ret = {
            'M': x1,
            'A': x2,
            'V': x3,
        }
        return ret
        
    def forward_feature_1(self, x_audio, x_visual, tau):
        # --fusion gs_it：交叉注意力 + Gumbel 采样融合（完整 G²AF）
        return self._two_stream(x_audio, x_visual, gumbel=True, tau=tau)

    def forward_feature_4(self, x_audio, x_visual, tau):
       # 1. Audio Feature Extraction
       x_audio = self.audio_model.forward_hetero(x_audio)

       # 2. Visual Feature Extraction
       x_visual = self.visual_model.forward_features(x_visual)
       x_visual = self.visual_model.forward_stage1(x_visual)
    
       # 3. Cross-Modal Interaction (Optional, keep your original logic)
       x_visual_ = x_visual.permute(0, 2, 1).mean([-2]) #(batchsize,64)
       x_audio_ = x_audio.permute(0, 2, 1).mean([-2]) #(batchsize,128)

       proj_x_a = x_audio.permute(0, 2, 1) #(batchsize,150,128)
       proj_x_v = x_visual.permute(0, 2, 1) #(batchsize,15,64)
       
       h_av = self._attn_matrix(self.av_blocks, proj_x_v, proj_x_a) # (batchsize,head,150,15)
       h_va = self._attn_matrix(self.va_blocks, proj_x_a, proj_x_v) # (batchsize,head,15,150)
       
       # 单向时 h_av / h_va 可能是 None（该方向被关掉）
       if h_av is not None:
           if h_av.size(1) > 1:
               h_av = torch.mean(h_av, axis=1).unsqueeze(1)
           h_av = h_av.sum([-2])
       if h_va is not None:
           if h_va.size(1) > 1:
               h_va = torch.mean(h_va, axis=1).unsqueeze(1)
           h_va = h_va.sum([-2])

       if h_va is not None:
           x_audio = h_va*x_audio
       if h_av is not None:
           x_visual = h_av*x_visual
        
       x_audio = self.audio_model.forward_stage2(x_audio)       
       x_visual = self.visual_model.forward_stage2(x_visual)

       audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
       video_pooled = x_visual.mean([-1])
       # =============================================
       # 5. Replace concat with Sampling Fusion
       # =============================================
       # Step 1: Feature Transformation
       f_sem = audio_pooled  # Treat audio as "semantic" feature
       f_spe = video_pooled  # Treat visual as "spectral" feature

       #维度匹配的特征变换 Linear layers to double dimensions 
       f_sem_transformed = torch.relu(self.fusion_linear_sem(f_spe))  
       f_spe_transformed = torch.relu(self.fusion_linear_spe(f_sem)) 

       # Element-wise multiplication
       f_sem_prime = f_sem * f_sem_transformed
       f_spe_prime = f_spe * f_spe_transformed

       # Step 2: Gumbel-Softmax Sampling
       s_sem = self.gumbel_sigmoid(f_sem, tau=tau, hard=False)
       s_spe = self.gumbel_sigmoid(f_spe, tau=tau, hard=False)

       # Step 3: Final Fusion 残差连接提升稳定性
       h_sem = f_sem_prime * s_sem + f_sem
       h_spe = f_spe_prime * s_spe + f_spe

       # Concatenate the fused features (now more robust than naive concat)
       x = torch.cat([h_spe, h_sem], dim=-1)
       # =============================================

       # 6. Classification Heads
       x1 = self.classifier_1(x)       # Fused modality
       x2 = self.classifier_2(x_audio_) # Audio-only
       x3 = self.classifier_3(x_visual_) # Visual-only

       return {'M': x1, 'A': x2, 'V': x3}


 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
