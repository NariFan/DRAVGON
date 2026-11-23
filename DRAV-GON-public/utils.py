'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import torch
import shutil
import numpy as np
import sklearn
from datetime import datetime
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a') #使用 'a' 模式打开文件
        self.logger = csv.writer(self.log_file, delimiter='\t')
        # 如果文件是新创建的，写入表头
        if os.path.getsize(path) == 0:
            self.logger.writerow(header)
        else:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.logger.writerow(['------', now, '-----------'])
            self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            if col not in values:
                print('col:',col,' not in values',',values:',values,',self.header:',self.header)
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    
    #print("Output shape:", output.shape)  # 应为 [batch_size, 3]
    #print("Target shape:", target.shape)  # 应为 [batch_size]
    #print("Target values:", target)       # 确保值在 [0, 1, 2] 范围内
    
    # 强制同步 CUDA 操作，确保错误立即暴露
    if output.is_cuda:
        torch.cuda.synchronize()
    
    maxk = max(topk)
    batch_size = target.size(0)
    
    # 确保 target 是 1D 且为整数类型
    target = target.view(-1).long()  # 关键修复！
    
    # 处理 maxk 超过输出维度的情况
    if maxk > output.size(1):
        maxk = output.size(1)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    # 检查维度
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    if binary:
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        return res, f1*100
    
    return res

def safe_torch_load(path, map_location=None):
    """
    兼容不同 PyTorch 版本的权重加载。

    背景：从 torch 2.6 开始，torch.load 的默认参数变成了 weights_only=True，
    出于安全考虑只允许加载张量。而本项目的 checkpoint 里除了张量还有
    str / int / dict，BEATs 权重里甚至有 BEATsConfig 这样的自定义对象，
    用默认的严格模式会直接抛 UnpicklingError。

    这里显式指定 weights_only=False。老版本 PyTorch（<1.13）不认这个关键字，
    会抛 TypeError，所以再用一次不带该参数的调用兜底。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 老版本 torch 没有 weights_only 参数
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# 实验命名
# ---------------------------------------------------------------------------
# 以前 store_name 是把 --store_name、opt.arch、opt.dataset、sample_duration 拼起来的，
# 结果是这种既带空格又自相矛盾的文件名：
#     gs_ia-1-noise-strat30f-e2v_model:multimodalcnn fusion:gs_ia num_heads:1_RAVDESS_30_best2.pth
# 其中 RAVDESS 是错的 —— dataset.py 里 assert 只允许 'RAVDESS'，PKU-ER 实验被迫
# 顶着 RAVDESS 的名字跑。
#
# 现在改成：实验名完全由「配置」推导，人不可能写错，也不可能漏掉某个消融轴。
# 目录布局也一并改成一实验一目录、一折一子目录：
#
#   <result_path>/<实验名>/fold0/{best.pth, checkpoint.pth, train.log, val.log, test.log}
#

_DATASET_ALIAS = {
    'ravdess': 'ravdess', 'rav': 'ravdess',
    'pku': 'pku', 'pku-er': 'pku', 'pku_er': 'pku', 'pkuer': 'pku',
}

_DIP_ALIAS = {'noise': 'nit', 'softhard': 'hm', 'nodropout': 'odt'}

_BACKBONE_ALIAS = {'beats': 'beats', 'emotion2vec': 'e2v'}


def dataset_tag(opt):
    """数据集名 -> 命名用的短标签，并做一次类别数一致性检查。"""
    raw = str(opt.dataset).strip().lower()
    tag = _DATASET_ALIAS.get(raw, raw.replace('_', '-'))
    # 两个数据集的类别数是固定的：RAVDESS 8 类、PKU-ER 3 类。
    # 不一致几乎肯定是 --dataset 忘了改，直接大声警告，不做静默猜测。
    if tag == 'ravdess' and opt.n_classes == 3:
        print('[WARN] dataset=RAVDESS 但 n_classes=3（PKU-ER 的配置）。请显式传 --dataset PKU-ER')
    if tag == 'pku' and opt.n_classes == 8:
        print('[WARN] dataset=PKU-ER 但 n_classes=8（RAVDESS 的配置）。请显式传 --dataset RAVDESS')
    return tag


def build_store_name(opt):
    """
    由完整配置推导出唯一、可读、可排序的实验名。

    四段固定顺序，每段内部用 '-' 连接、段之间用 '_' 连接：

        ravdess-30f-e2v_gs_it-h1-nit_har-full_dir-both-d1
        |_____ 数据 ____| |___ 结构 ___| | 声学 | | 交互方向 |

    例：
        ravdess-30f-e2v_gs_it-h1-nit_har-full_dir-both-d1   完整模型
        ravdess-30f-e2v_gs_it-h1-nit_har-deep_dir-both-d1   w/o HAR-Learner
        ravdess-30f-e2v_it-h1-nit_har-full_dir-both-d1      w/o Gumbel 融合
        ravdess-30f-e2v_concat-h1-nit_har-full_dir-both-d1  w/o 整个 G²AF
        ravdess-30f-e2v_gs_it-h1-odt_har-full_dir-both-d1   w/o DIP
        ravdess-30f-e2v_gs_it-h1-nit_har-full_dir-av-d2     单向（容量匹配）
    """
    seg_data = '%s-%df-%s' % (
        dataset_tag(opt),
        opt.sample_duration,
        _BACKBONE_ALIAS.get(str(opt.audio_backbone).lower(), opt.audio_backbone),
    )
    seg_arch = '%s-h%d-%s' % (
        opt.fusion,
        opt.num_heads,
        _DIP_ALIAS.get(str(opt.mask).lower(), opt.mask),
    )
    seg_har = 'har-%s' % opt.har
    depth = 1 if opt.cross_dir == 'both' else int(opt.cross_depth)
    seg_dir = 'dir-%s-d%d' % (opt.cross_dir, depth)

    name = '_'.join([seg_data, seg_arch, seg_har, seg_dir])
    if getattr(opt, 'tag', ''):
        name = '%s_%s' % (name, opt.tag)
    return name


def build_arch(opt):
    """写入 checkpoint 的结构签名。加载 resume_path 时用来校验配置没变。"""
    return 'model:%s fusion:%s num_heads:%d har:%s cross_dir:%s cross_depth:%d' % (
        opt.model, opt.fusion, opt.num_heads,
        getattr(opt, 'har', 'full'), getattr(opt, 'cross_dir', 'both'),
        1 if getattr(opt, 'cross_dir', 'both') == 'both' else int(getattr(opt, 'cross_depth', 1)),
    )


def exp_dir(opt, fold=None):
    """
    实验目录。fold 为 None 时返回实验根目录，否则返回该折的子目录。

        实验根  = <result_path>/<实验名>
        某一折  = <result_path>/<实验名>/fold<N>
    """
    d = os.path.join(opt.result_path, opt.store_name)
    return d if fold is None else os.path.join(d, 'fold%d' % fold)


def save_checkpoint(state, is_best, opt, fold):
    """
    存到 <实验根>/fold<N>/ 下，文件名固定为 checkpoint.pth / best.pth。
    折号已经体现在目录里，不再拼进文件名，避免 _best0 / _best2 这种后缀。
    """
    d = exp_dir(opt, fold)
    os.makedirs(d, exist_ok=True)
    ckpt = os.path.join(d, 'checkpoint.pth')
    best = os.path.join(d, 'best.pth')
    torch.save(state, ckpt)
    if is_best:
        shutil.copyfile(ckpt, best)


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate


