# -*- coding: utf-8 -*-
'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', default='ravdess_preprocessing/annotations.txt', type=str, help='Annotation file path')
    parser.add_argument('--result_path', default='results06', type=str, help='Result directory path')
    # store_name 不再由命令行拼出，而是由 utils.build_store_name(opt) 依据完整配置
    # 自动生成（见下方「实验命名」小节）。--tag 只是可选的可读后缀，追加在末尾。
    parser.add_argument('--tag', default='', type=str,
                        help='可选。追加在自动生成的实验名末尾的自定义后缀')
    # 真实数据集名，用于实验命名与目录划分。
    # RAVDESS 与 PKU-ER 共用同一套 loader（标注格式相同），所以改成 PKU-ER 不会
    # 改变加载逻辑，只会让实验名里的 dataset 段正确地显示为 pku 而不是 ravdess。
    parser.add_argument('--dataset', default='RAVDESS', type=str,
                        help='Used dataset: RAVDESS | PKU-ER。仅影响命名，两者共用 loader')
    parser.add_argument('--n_classes', default=8, type=int, help='Number of classes')
    
    parser.add_argument('--model', default='multimodalcnn', type=str, help='')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')
    
    
    parser.add_argument('--sample_size', default=224, type=int, help='Video dimensions: ravdess = 224 ')
    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 15')
    
    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')

    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')
    
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='', type=str, help='Save data (.pth) of previous training')

    parser.add_argument('--pretrain_path', default='Pretrain-model/EfficientFace_Trained_on_AffectNet7.pth.tar', type=str, help='Pretrained model (.pth), efficientface')

    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)

    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')
    
    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--video_norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
 
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--fusion', default='ia', type=str,
                        help='fusion type: concat | lt | it | ia | gs_it | gs_lt | gs_ia')
    # 训练时对训练集数据进行的处理
    parser.add_argument('--mask', type=str, help='dropout type : softhard | noise | nodropout', default='noise')
    # 测试时对测试集数据进行的处理
    parser.add_argument('--test_val_modality', type=str, help='modality type : both | audio | video 在进行测试时对哪个模态引入噪声，both代表对两个模态都不处理', default='both')
    parser.add_argument('--test_val_modality_dist', type=str,
                        help='dist type : noise | addnoise | zeros ;当test_val_modality设置为audio或者video的时候dist决定引入噪声的方式', default='noise')

    # ------------------------------------------------------------------
    # 以下参数是为回应审稿人意见、补充对照实验而新增的（2026 年修回版）。
    # 所有默认值都刻意设置成与修改前完全一致，因此即使不加任何新参数，
    # 原有的训练/测试流程与结果也不会改变。
    # ------------------------------------------------------------------

    # ---- 问题2：K 折交叉验证 ----
    # n_folds=1 时行为与以前完全相同（只读 annotation_path 本身）。
    # n_folds>1 时，main.py 会把标注路径改写为 <annotation_path 去掉 .txt>_<fold>.txt，
    # 所以需要先用 ravdess_preprocessing/create_annotations_5fold.py 生成这批折文件。
    parser.add_argument('--n_folds', default=1, type=int,
                        help='交叉验证折数。1 表示不交叉（保持原行为）；5 表示 5 折交叉验证')
    # 只跑指定折（多路并行 K 折时用）。逗号分隔，如 "0,1,2"；留空 = 跑全部 n_folds 折。
    # 例：进程A --fold_list 0,1,2，进程B --fold_list 3,4，两路并行跑完 5 折。
    parser.add_argument('--fold_list', default='', type=str,
                        help='只跑这些折（逗号分隔），留空表示跑全部折')

    # ---- 问题1：Gumbel 模块超参数 ----
    # Gumbel 采样融合的温度 tau。训练时按线性退火：
    #     tau = max(tau_start - (epoch*10 + batch) * tau_decay, tau_end)
    # 默认值 1.0 / 0.1 / 0.001 与修改前 train.py 里的硬编码公式完全等价。
    # 【重要】models/multimodalcnn.py 的 gumbel_sigmoid 内部有 clamp(tau, min=0.1, max=1.0)，
    # 因此 tau 小于 0.1 会被静默钳到 0.1，实验曲线会变成一条平线。
    # 做超参数扫描时取值请保持在 [0.1, 1.0] 区间内，否则扫不出差异。
    parser.add_argument('--tau_start', default=1.0, type=float, help='Gumbel 温度初始值')
    parser.add_argument('--tau_end', default=0.1, type=float, help='Gumbel 温度下限（退火终点）')
    parser.add_argument('--tau_decay', default=0.001, type=float, help='Gumbel 温度每步衰减量')
    parser.add_argument('--eval_tau', default=0.01, type=float,
                        help='验证/测试时使用的 Gumbel 温度（原代码硬编码 0.01，注意会被 clamp 到 0.1）')

    # ---- 问题3：多梯度信噪比噪声鲁棒性 ----
    # test_snr 为 None 表示不加噪（干净测试集）；给数值则按该信噪比(dB)加高斯白噪声。
    # 噪声加在波形层，即读取 wav 之后、提取 MFCC/CQT/BEATs 之前，见 datasets/ravdess.py。
    parser.add_argument('--test_snr', default=None, type=float,
                        help='测试集信噪比(dB)，数值越小噪声越强。留空表示不加噪')
    parser.add_argument('--snr_power_mode', default='active', type=str,
                        help="信噪比口径：active=按有声帧功率 | full=按整段波形功率")
    parser.add_argument('--snr_top_db', default=20, type=int,
                        help='有声帧检测阈值(dB)，仅 snr_power_mode=active 时生效')
    parser.add_argument('--noise_seed', default=1234, type=int,
                        help='噪声随机种子。按「文件名+信噪比」播种，保证结果可复现')

    # ---- 通用实验便利开关 ----
    # test_only_both=True 时，只跑「两个模态都不破坏」这一组测试，
    # 跳过原来固定执行的另外 6 个模态破坏组合。做扫描实验时能省掉 6/7 的测试时间。
    parser.add_argument('--test_only_both', action='store_true',
                        help='只在未破坏模态下测试，跳过其余 6 个模态破坏组合以节省时间')
    parser.set_defaults(test_only_both=False)

    # summary_csv 非空时，每折结束往该文件追加一行结构化结果，供 plot_all.py 直接读取作图。
    parser.add_argument('--summary_csv', default='', type=str,
                        help='结构化结果汇总 CSV 路径，留空表示不输出')

    # BEATs 权重路径。原代码在 datasets/ravdess.py 里硬编码，这里提出来方便不同服务器配置。
    # 默认值与原来完全一致，因此不加这个参数时行为不变。
    parser.add_argument('--beats_ckpt', default='./pre-trained_checkpoints/BEATs_iter3.pt', type=str,
                        help='BEATs 预训练权重路径')

    # BEATs 特征缓存目录。留空（默认）表示每次都重新推理，行为与以前一致。
    # 填了目录后，dataset 在「不加噪」时会直接读预提取好的特征，训练速度提升数倍。
    # 加噪场景永远走在线计算，不受这个参数影响。
    parser.add_argument('--beats_cache_dir', default='', type=str,
                        help='预提取的 BEATs 特征缓存目录，留空表示不用缓存')

    # 音频基座选择（审稿人 C1 对照实验）。默认 beats，行为与以前完全一致。
    # emotion2vec 是情感专用 SSL 预训练模型，用于验证 domain bias 的影响。
    # 注意：换基座时 --beats_cache_dir 必须指向「同一基座」提取的缓存目录。
    parser.add_argument('--audio_backbone', default='beats', type=str,
                        choices=['beats', 'emotion2vec'],
                        help='音频基座：beats（默认）或 emotion2vec')
    parser.add_argument('--emotion2vec_dir', default='./pre-trained_checkpoints/emotion2vec', type=str,
                        help='emotion2vec 权重目录')

    # ------------------------------------------------------------------
    # 消融实验开关（回应审稿人「Table A1 只做了单折、且组件被联合去掉」的意见）。
    # 三个开关的默认值都等于完整模型，不加任何参数时行为与以前完全一致。
    # 每个开关都会出现在自动生成的实验名里，见 utils.build_store_name。
    # ------------------------------------------------------------------

    # HAR-Learner：声学特征的「异构」程度。
    #   full         = MFCC(10) + CQT(42) + 深度特征(768)      <- 完整 HAR-Learner
    #   deep         = 只有深度特征，去掉手工频谱特征            <- w/o HAR-Learner
    #   handcrafted  = 只有 MFCC + CQT，去掉深度特征            <- 反向对照
    parser.add_argument('--har', default='full', type=str,
                        choices=['full', 'deep', 'handcrafted'],
                        help='HAR-Learner 声学特征组成')

    # 交叉注意力方向，用于「双向 vs 单向」对照。
    #   both = 双向（av + va），即论文里的结构性对称交互
    #   av   = 单向：音频查询视觉（只有音频被视觉增强，视觉不被音频更新）
    #   va   = 单向：视觉查询音频（只有视觉被音频增强，音频不被视觉更新）
    parser.add_argument('--cross_dir', default='both', type=str,
                        choices=['both', 'av', 'va'],
                        help='交叉注意力方向')

    # 单向时把同一方向的 block 堆叠 N 层，用来做「容量匹配」：
    # 朴素单向只有双向的 74%(av) / 26%(va) 参数，赢了也可能只是参数量差异所致。
    # gs_it 下实测：av 单层 115.8k、va 单层 41.7k、双向合计 157.5k。
    # 取 av×2=231.6k / va×4=166.8k 即可让单向的交叉注意力参数反超双向。
    # cross_dir=both 时本参数恒为 1，填其他值会被忽略。
    parser.add_argument('--cross_depth', default=1, type=int,
                        help='单向交叉注意力的堆叠层数（容量匹配用）')

    args = parser.parse_args()

    return args
