'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger,awl):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_ = AverageMeter()
    loss1_ = AverageMeter()
    loss2_ = AverageMeter()
    loss3_ = AverageMeter()
    awl1_ = AverageMeter()
    awl2_ = AverageMeter()
    awl3_ = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    top1_a = AverageMeter()
    top5_a = AverageMeter()
    top1_v = AverageMeter()
    top5_v = AverageMeter()
        
    end_time = time.time()

    # 打印模型推理图的各个叶子节点
    if epoch == 2:
        print('打印模型推理图的各个叶子节点')
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Gradient of {name}: {param.grad.shape}')

    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

   
        # 先把干净 batch 搬到 GPU（体积只有 batch 本身），
        # 再在 GPU 上做 DIP 增强：噪声/系数随机张量、cat、打乱全部走 CUDA。
        # 原来的实现是在 CPU 上 cat 成 3~4 倍大之后再整体传 GPU（视觉一批就 217MB），
        # 这一步既耗 CPU 又拖慢 PCIe 传输，是每个 epoch 的大头。
        audio_inputs = audio_inputs.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)
        targets = targets.to(opt.device)
            
        if opt.mask is not None:
            with torch.no_grad():
                
                if opt.mask == 'noise':
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size(), device=opt.device), audio_inputs), dim=0)                   
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size(), device=opt.device)), dim=0) 
                    targets = torch.cat((targets, targets, targets), dim=0)                    
                    shuffle = torch.randperm(audio_inputs.size()[0], device=opt.device)
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    
                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0),1,1), device=opt.device) / 100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

                    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size(), device=opt.device), audio_inputs), dim=0) 
                    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size(), device=opt.device)), dim=0)   
                    
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0], device=opt.device)
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
   
  

        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)

        targets = Variable(targets)
        
        # Gumbel 采样融合的温度，线性退火：tau_start → tau_end
        # （回应审稿人问题1：把 Gumbel 温度做成可从命令行调节的超参数）
        # 默认取值 1.0 / 0.001 / 0.1 与修改前这里的硬编码公式完全等价。
        #
        # 【重要提醒】models/multimodalcnn.py 的 gumbel_sigmoid 内部做了
        # clamp(tau, min=0.1, max=1.0)，所以 tau_end 设成小于 0.1 的值不会生效，
        # 会被静默钳到 0.1，扫描实验会得到一条平线。扫描请取 [0.1, 1.0]。
        tau = max(
            getattr(opt, 'tau_start', 1.0) - (epoch * 10 + i) * getattr(opt, 'tau_decay', 0.001),
            getattr(opt, 'tau_end', 0.1))
        # 多任务训练方法
        outputs = model(audio_inputs, visual_inputs, tau)
        outputs_av = outputs['M']
        outputs_a = outputs['A']
        outputs_v = outputs['V']
        #print('outputs_v:',outputs_v.size(), ' targets:',targets.size(),' targets:',targets)
        loss1 = criterion(outputs_av, targets)
        loss2 = criterion(outputs_a, targets)
        loss3 = criterion(outputs_v,targets)


        # loss = 3 * loss1 + loss2 + loss3
        # losses = loss1_task_weight*loss1 + loss2_task_weight*loss2 + loss3_task_weight*loss3
        losses = awl(loss1,loss2,loss3)

        losses_.update(losses.data, audio_inputs.size(0))
        loss1_.update(loss1.data, audio_inputs.size(0))
        loss2_.update(loss2.data, audio_inputs.size(0))
        loss3_.update(loss3.data, audio_inputs.size(0))
        # for p in awl.parameters():
        #     print(p.shape)
        awl_params = [p.tolist() for p in awl.parameters()]  # 获取参数并转换为浮点数列表
        # print('awl:',awl_params)
        awl1_.update(awl_params[0][0],1)
        awl2_.update(awl_params[0][1],1)
        awl3_.update(awl_params[0][2],1)

        prec1, prec5 = calculate_accuracy(outputs_av.data, targets.data, topk=(1,2))
        prec1_a, prec5_a = calculate_accuracy(outputs_a.data, targets.data, topk=(1, 2))
        prec1_v, prec5_v = calculate_accuracy(outputs_v.data, targets.data, topk=(1, 2))

        top1_m.update(prec1, audio_inputs.size(0))
        top5_m.update(prec5, audio_inputs.size(0))
        top1_a.update(prec1_a, audio_inputs.size(0))
        top5_a.update(prec5_a, audio_inputs.size(0))
        top1_v.update(prec1_v, audio_inputs.size(0))
        top5_v.update(prec5_v, audio_inputs.size(0))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'losses': losses_.val.item(),
            'loss1': loss1_.val.item(),
            'loss2': loss2_.val.item(),
            'loss3': loss3_.val.item(),
            'awl1': awl1_.val,
            'awl2': awl2_.val,
            'awl3': awl3_.val,
            'prec1_m': top1_m.val.item(),
            'prec5_m': top5_m.val.item(),
            'prec1_a': top1_a.val.item(),
            'prec5_a': top5_a.val.item(),
            'prec1_v': top1_v.val.item(),
            'prec5_v': top5_v.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Losses {losses_.val:.4f} ({losses_.avg:.4f})\t'
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
                      losses_=losses_,
                      loss1_=loss1_,
                      loss2_=loss2_,
                      loss3_=loss3_,
                      top1_m=top1_m,
                      top5_m=top5_m,
                      top1_a=top1_a,
                      top5_a=top5_a,
                      top1_v=top1_v,
                      top5_v=top5_v,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'losses': losses_.avg.item(),
        'loss1' : loss1_.avg.item(),
        'loss2': loss2_.avg.item(),
        'loss3': loss3_.avg.item(),
        'prec1_m': top1_m.avg.item(),
        'prec5_m': top5_m.avg.item(),
        'prec1_a': top1_a.avg.item(),
        'prec5_a': top5_a.avg.item(),
        'prec1_v': top1_v.avg.item(),
        'prec5_v': top5_v.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger,awl=None):
    print('train at epoch {}'.format(epoch))
    
    if opt.model == 'multimodalcnn':
        train_epoch_multimodal(epoch,  data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger,awl)
        return
    
    
