'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import os
import csv

import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy


# 逐样本预测落盘的表头（回应审稿人问题9）。
# 这里只记录「原始观测量」：真实标签与三个头的预测类别、以及融合头的置信度。
# 正确与否（cm/ca/cv）、错分类型（C1~C6）等派生量一律留给 analyze_stats.py 离线计算，
# 这样同一份 dump 可以用不同的口径反复统计，也便于审稿人复查。
_DUMP_FIELDS = ['index', 'label', 'pred_m', 'pred_a', 'pred_v', 'conf_m',
                'audio_path', 'video_path']


def _write_dump(dump_path, rows, snr_tag, fold=None):
    """
    把逐样本预测写成 CSV，供 analyze_stats.py / plot_all.py 读取。

    参数
    ----
    dump_path : 目标 CSV 路径；所在目录不存在时自动创建
    rows      : 每行一个样本的字典列表
    snr_tag   : 信噪比标签，'clean' 或数值字符串，写进 snr_db 列
    fold      : 折号，可为空

    说明：这里做的是覆盖写（'w'），而不是追加。
    因为同一次评估只应产出一份结果，追加会在重跑时造成行重复、统计虚高。
    """
    if not rows:
        return
    directory = os.path.dirname(os.path.abspath(dump_path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    fieldnames = ['fold', 'snr_db'] + _DUMP_FIELDS
    with open(dump_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, '') for k in _DUMP_FIELDS}
            out['fold'] = '' if fold is None else fold
            out['snr_db'] = snr_tag
            writer.writerow(out)


def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger,
                         modality='both', dist=None,
                         # ↓↓↓ 审稿人补充实验新增的可选参数，不传即保持原有行为 ↓↓↓
                         dump_path=None, snr_tag='clean', fold=None):
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']    
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss1_ = AverageMeter()
    loss2_ = AverageMeter()
    loss3_ = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    top1_a = AverageMeter()
    top5_a = AverageMeter()
    top1_v = AverageMeter()
    top5_v = AverageMeter()

    # 逐样本预测收集容器。仅当调用方传入 dump_path 时才填充，否则保持为空、零开销。
    dump_rows = []

    # 整个 epoch 的预测收集器。Balanced Accuracy 必须在全数据集上按类统计
    # recall 再平均，不能按 batch 累加。原实现把这几个收集器放在循环体内，
    # 每个批次都被清空成只剩最后一批 —— 顺带修掉。
    all_preds = []
    all_preds_A = []
    all_preds_V = []
    all_targets = []

    end_time = time.time()
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        '''
        检查模态（modality）：
        
        如果模态是audio，则跳过视频模态的处理。
        如果模态是video，则跳过音频模态的处理。
        根据扰动类型（dist）处理输入数据：
        
        对于视频模态（modality == 'audio'）：
        如果扰动类型是noise，则用全噪声替换视频输入数据。
        如果扰动类型是addnoise，则在视频输入数据上添加噪声。
        如果扰动类型是zeros，则用零替换视频输入数据。
        如果扰动类型未知，打印UNKNOWN DIST!。
        对于音频模态（modality == 'video'）：
        如果扰动类型是noise，则用全噪声替换音频输入数据。
        如果扰动类型是addnoise，则在音频输入数据上添加噪声。
        如果扰动类型是zeros，则用零替换音频输入数据。
        处理视频输入数据的形状：
        
        将视频输入数据的维度重新排列（permute）。
        将视频输入数据展平（reshape）为一维数组。
        '''
        if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual)*torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print('UNKNOWN DIST!')
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio)*torch.randn(inputs_audio.size()))

            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())

        inputs_visual = inputs_visual.permute(0,2,1,3,4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])

        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)

        # Gumbel 温度。原来是硬编码的 0.01，现改为可配（回应审稿人问题1）。
        # 默认值 0.01 与修改前完全一致。
        # 注意：gumbel_sigmoid 内部有 clamp(tau, min=0.1, max=1.0)，
        # 所以传入 0.01 实际会被钳到 0.1 —— 这与训练时退火终点 0.1 恰好一致，
        # 是原代码能正常工作的隐含前提，改动此值时请同步考虑训练侧的 tau_end。
        tau = getattr(opt, 'eval_tau', 0.01)
        outputs = model(inputs_audio, inputs_visual, tau)
        outputs_av = outputs['M']
        outputs_a = outputs['A']
        outputs_v = outputs['V']
        loss1 = criterion(outputs_av, targets)
        loss2 = criterion(outputs_a, targets)
        loss3 = criterion(outputs_v, targets)
        loss = loss1 + loss2 + loss3
        
        # 收集预测值和目标值（供 Balanced Accuracy 与逐样本落盘使用）
        preds = outputs_av.argmax(dim=1)  # 融合头 M 的预测
        preds_a = outputs_a.argmax(dim=1)
        preds_v = outputs_v.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_preds_A.extend(preds_a.cpu().numpy().tolist())
        all_preds_V.extend(preds_v.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
    
        # === 逐样本预测落盘（回应审稿人问题9）===
        # 只在调用方要求时才收集。样本下标 = 批内偏移 + 批序号×批大小，
        # 这依赖于 DataLoader 的 shuffle=False（测试/验证加载器都是如此设置）。
        # 若开启 shuffle，下标将无法对应回标注文件里的原始样本，
        # 因此这里再尽力从 dataset 里取回真实路径作为交叉校验。
        if dump_path is not None:
            # 融合头 M 的置信度（最大 softmax 概率），用于后续按置信度分层分析
            conf_m = torch.softmax(outputs_av, dim=1).max(dim=1).values
            dataset_samples = getattr(getattr(data_loader, 'dataset', None), 'data', None)
            batch_base = i * targets.size(0)
            for b in range(targets.size(0)):
                idx = batch_base + b
                row = {
                    'index': idx,
                    'label': int(targets[b].item()),
                    'pred_m': int(preds[b].item()),
                    'pred_a': int(preds_a[b].item()),
                    'pred_v': int(preds_v[b].item()),
                    'conf_m': float(conf_m[b].item()),
                    'audio_path': '',
                    'video_path': '',
                }
                if dataset_samples is not None and idx < len(dataset_samples):
                    row['audio_path'] = dataset_samples[idx].get('audio_path', '')
                    row['video_path'] = dataset_samples[idx].get('video_path', '')
                dump_rows.append(row)

        prec1, prec5 = calculate_accuracy(outputs_av.data, targets.data, topk=(1, 2))
        prec1_a, prec5_a = calculate_accuracy(outputs_a.data, targets.data, topk=(1, 2))
        prec1_v, prec5_v = calculate_accuracy(outputs_v.data, targets.data, topk=(1, 2))
        top1_m.update(prec1, inputs_audio.size(0))
        top5_m.update(prec5, inputs_audio.size(0))
        top1_a.update(prec1_a, inputs_audio.size(0))
        top5_a.update(prec5_a, inputs_audio.size(0))
        top1_v.update(prec1_v, inputs_audio.size(0))
        top5_v.update(prec5_v, inputs_audio.size(0))
        # prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        # top1.update(prec1, inputs_audio.size(0))
        # top5.update(prec5, inputs_audio.size(0))

        losses.update(loss.data, inputs_audio.size(0))
        loss1_.update(loss1.data, inputs_audio.size(0))
        loss2_.update(loss2.data, inputs_audio.size(0))
        loss3_.update(loss3.data, inputs_audio.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Loss1 {loss1_.val:.4f} ({loss1_.avg:.4f})\t'
              'Loss2 {loss2_.val:.4f} ({loss2_.avg:.4f})\t'
              'Loss3 {loss3_.val:.4f} ({loss3_.avg:.4f})\t'
              'Prec_m@1 {top1_m.val:.5f} ({top1_m.avg:.5f})\t'
              'Prec_m@5 {top5_m.val:.5f} ({top5_m.avg:.5f})\t'
              'Prec_a@1 {top1_a.val:.5f} ({top1_a.avg:.5f})\t'
              'Prec_a@5 {top5_a.val:.5f} ({top5_a.avg:.5f})\t'
              'Prec_v@1 {top1_v.val:.5f} ({top1_v.avg:.5f})\t'
              'Prec_v@5 {top5_v.val:.5f} ({top5_v.avg:.5f})\t'
            .format(
            epoch,
            i,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            loss1_=loss1_,
            loss2_=loss2_,
            loss3_=loss3_,
            top1_m=top1_m,
            top5_m=top5_m,
            top1_a=top1_a,
            top5_a=top5_a,
            top1_v=top1_v,
            top5_v=top5_v))

    # === Balanced Accuracy（各类 recall 的平均）===
    # 回应审稿人 2 意见 C1：PKU-ER 三类样本数并不相等（288/291/323），
    # 总体 accuracy 会被多数类主导，要求补充该指标。
    from sklearn.metrics import balanced_accuracy_score
    if all_targets:
        bacc_m = balanced_accuracy_score(all_targets, all_preds) * 100.0
        bacc_a = balanced_accuracy_score(all_targets, all_preds_A) * 100.0
        bacc_v = balanced_accuracy_score(all_targets, all_preds_V) * 100.0
    else:  # 数据加载器为空的边界情况（例如配置错误的折文件）
        bacc_m = bacc_a = bacc_v = 0.0

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'loss1': loss1_.avg.item(),
                'loss2': loss2_.avg.item(),
                'loss3': loss3_.avg.item(),
                'prec1_m': top1_m.avg.item(),
                'prec5_m': top5_m.avg.item(),
                'prec1_a': top1_a.avg.item(),
                'prec5_a': top5_a.avg.item(),
                'prec1_v': top1_v.avg.item(),
                'prec5_v': top5_v.avg.item(),
                'bacc_m': bacc_m,
                'bacc_a': bacc_a,
                'bacc_v': bacc_v
                })

    # 逐样本预测落盘。放在最后统一写，避免每个 batch 都开一次文件。
    if dump_path is not None:
        _write_dump(dump_path, dump_rows, snr_tag=snr_tag, fold=fold)
        print('逐样本预测已写入: {} (共 {} 条)'.format(dump_path, len(dump_rows)))

    return (losses.avg.item(), top1_m.avg.item(), top1_a.avg.item(), top1_v.avg.item(),
            bacc_m, bacc_a, bacc_v)

def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None,
              dump_path=None, snr_tag='clean', fold=None):
    print('validation at epoch {}'.format(epoch))
    if opt.model == 'multimodalcnn':
        return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist,
                                    dump_path=dump_path, snr_tag=snr_tag, fold=fold)
    
