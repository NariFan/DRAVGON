# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import csv
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import (Logger, adjust_learning_rate, save_checkpoint, safe_torch_load,
                   build_store_name, build_arch, exp_dir, dataset_tag)
from train import train_epoch
from validation import val_epoch
import time
from AutomaticWeightedLoss import AutomaticWeightedLoss

# 结构化结果汇总的表头（供 plot_all.py 直接读取作图）
SUMMARY_FIELDS = ['store_name', 'dataset', 'frames', 'backbone', 'fusion', 'num_heads',
                   'dip', 'har', 'cross_dir', 'cross_depth',
                   'tau_start', 'tau_end', 'eval_tau',
                   'fold', 'prec1_m', 'prec1_a', 'prec1_v', 'bacc_m', 'bacc_a', 'bacc_v']
_SUMMARY_FIELDS = SUMMARY_FIELDS  # 兼容旧名


def _append_summary(opt, fold, prec1_m, prec1_a, prec1_v, bacc_m=0, bacc_a=0, bacc_v=0):
    """
    往 --summary_csv 指定的 CSV 追加一行本折结果。

    Q1（Gumbel 温度扫描）会把不同温度的结果写进同一个 CSV，靠 tau_end/eval_tau 列区分；
    Q2（5 折交叉验证）会把 5 折写进同一个 CSV，靠 fold 列区分。
    两者共用一种表头，作图脚本因此只需要一套解析逻辑。

    路径规则：--summary_csv 是相对路径时，相对的是「运行 main.py 时的工作目录」，
    也就是项目根目录 DRAV-GON/。实验脚本里统一从根目录执行，所以直接写 q1_tau.csv 即可。
    """
    path = opt.summary_csv
    if not path:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'store_name': opt.store_name,
            'dataset': dataset_tag(opt),
            'frames': opt.sample_duration,
            'backbone': opt.audio_backbone,
            'fusion': opt.fusion,
            'num_heads': opt.num_heads,
            'dip': opt.mask,
            'har': getattr(opt, 'har', 'full'),
            'cross_dir': getattr(opt, 'cross_dir', 'both'),
            'cross_depth': 1 if getattr(opt, 'cross_dir', 'both') == 'both' else int(getattr(opt, 'cross_depth', 1)),
            'tau_start': getattr(opt, 'tau_start', 1.0),
            'tau_end': getattr(opt, 'tau_end', 0.1),
            'eval_tau': getattr(opt, 'eval_tau', 0.01),
            'fold': fold,
            'prec1_m': round(float(prec1_m), 4),
            'prec1_a': round(float(prec1_a), 4),
            'prec1_v': round(float(prec1_v), 4),
            'bacc_m': round(float(bacc_m), 4),
            'bacc_a': round(float(bacc_a), 4),
            'bacc_v': round(float(bacc_v), 4),
        })


if __name__ == '__main__':
    opt = parse_opts()
    # 交叉验证折数（回应审稿人问题2）。默认 1，即不做交叉验证，与修改前行为完全一致。
    # 设为 5 之前，请先用 ravdess_preprocessing/create_annotations_5fold.py 生成折文件，
    # 并把 --annotation_path 指向 .../annotations_fold.txt
    # （下面会自动改写成「去掉 .txt」+ '_' + 折号 + '.txt'）。
    n_folds = getattr(opt, 'n_folds', 1)
    # 只跑指定折（多路并行 K 折）。留空 = 全部折。
    fold_list = [int(x) for x in getattr(opt, 'fold_list', '').split(',') if x.strip() != '']
    folds_to_run = fold_list if fold_list else list(range(n_folds))
    if n_folds != 1:
        assert opt.annotation_path.endswith('.txt'), "annotation_path didn't end with .txt"
        annotation_path = opt.annotation_path
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    print('device',opt.device)

    pretrained = opt.pretrain_path != 'None'    
    
    #opt.result_path = 'res_'+str(time.time())
    # ---- 实验命名：完全由配置推导，人写不错也漏不掉消融轴 ----
    #   ravdess-30f-e2v_gs_it-h1-nit_har-full_dir-both-d1
    opt.arch = build_arch(opt)
    opt.store_name = build_store_name(opt)
    print('opt.arch      ', opt.arch)
    print('opt.store_name', opt.store_name)

    # 目录布局：<result_path>/<实验名>/fold<N>/{best.pth, checkpoint.pth, *.log}
    os.makedirs(opt.result_path, exist_ok=True)
    os.makedirs(exp_dir(opt), exist_ok=True)
    with open(os.path.join(exp_dir(opt), 'opts.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2)
    for fold in folds_to_run:
        #if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'
        if n_folds!=1:
            opt.annotation_path = annotation_path[:-4]+'_'+str(fold)+'.txt'
        print(opt)
        os.makedirs(exp_dir(opt, fold), exist_ok=True)
        with open(os.path.join(exp_dir(opt, fold), 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file, indent=2)
        # step1 :固定随机数
        torch.manual_seed(opt.manual_seed)
        np.random.seed(opt.manual_seed)
        torch.cuda.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed_all(opt.manual_seed)
        # step2 :设置模型结构
        model, parameters = generate_model(opt) #载入模型
        awl = AutomaticWeightedLoss(3)
        # step3 :设置loss函数结构
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)
        # step4 ：设置训练集、训练优化器、测试集
        if not opt.no_train:
            
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])
        
            training_data = get_training_set(opt, spatial_transform=video_transform) 
        
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True,
                persistent_workers=opt.n_threads > 0)
        
            train_logger = Logger(
                os.path.join(exp_dir(opt, fold), 'train.log'),
                ['epoch', 'losses', 'loss1', 'loss2', 'loss3', 'prec1_m', 'prec5_m', 'prec1_a', 'prec5_a', 'prec1_v', 'prec5_v', 'lr'])
            train_batch_logger = Logger(
                os.path.join(exp_dir(opt, fold), 'train_batch.log'),
                ['epoch', 'batch', 'iter', 'losses', 'loss1', 'loss2', 'loss3','awl1','awl2','awl3', 'prec1_m', 'prec5_m', 'prec1_a', 'prec5_a', 'prec1_v', 'prec5_v','lr'])
            train_logger.logger.writerow(
                ['------fold:',fold,'------fusion:', opt.fusion, '------num_heads:', opt.num_heads, '------mask:',opt.mask,'------'])

            # 没有使用AutomaticWeightedLoss进行多任务优化的方法
            # optimizer = optim.SGD(
            #     parameters,
            #     lr=opt.learning_rate,
            #     momentum=opt.momentum,
            #     dampening=opt.dampening,
            #     weight_decay=opt.weight_decay,
            #     nesterov=False)

            # 使用了AutomaticWeightedLoss进行多任务优化的方法
            optimizer = optim.SGD([
                {'params': model.parameters(), 'weight_decay': opt.weight_decay},
                {'params': awl.parameters(), 'weight_decay': 0}  # 或者你想要的任何其他设置
            ],
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
            
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])     
        
            validation_data = get_validation_set(opt, spatial_transform=video_transform)
            
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True,
                persistent_workers=opt.n_threads > 0)
        
            val_logger = Logger(
                    os.path.join(exp_dir(opt, fold), 'val.log'), ['epoch', 'loss','loss1', 'loss2', 'loss3', 'prec1_m', 'prec5_m', 'prec1_a', 'prec5_a', 'prec1_v', 'prec5_v', 'bacc_m', 'bacc_a', 'bacc_v'])
            test_logger = Logger(
                    os.path.join(exp_dir(opt, fold), 'test.log'), ['epoch', 'loss','loss1', 'loss2', 'loss3', 'prec1_m', 'prec5_m', 'prec1_a', 'prec5_a', 'prec1_v', 'prec5_v', 'bacc_m', 'bacc_a', 'bacc_v'])

            
        best_prec1 = 0
        best_loss = 1e10
        # step5 ：载入上一次训练的状态
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = safe_torch_load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        # step6 ：逐个epoch遍历，看看是否需要训练或验证
        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger,awl=awl)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt, fold)
            
            if not opt.no_val:
                
                validation_loss, prec1_m, prec1_a, prec1_v, _, _, _ = val_epoch(epoch=i, data_loader=val_loader, model=model, criterion=criterion, opt=opt,
                                            logger=val_logger)
                is_best = prec1_m > best_prec1
                best_prec1 = max(prec1_m, best_prec1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
               
                save_checkpoint(state, is_best, opt, fold)

        # step 7：测试
        if opt.test:

            # test_logger = Logger(
            #         os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger(
                os.path.join(exp_dir(opt, fold), 'test.log'),
                ['epoch', 'loss', 'loss1', 'loss2', 'loss3', 'prec1_m', 'prec5_m', 'prec1_a', 'prec5_a', 'prec1_v', 'prec5_v', 'bacc_m', 'bacc_a', 'bacc_v'])
            test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', opt.test_val_modality, '------test_val_modality_dist:',opt.test_val_modality_dist,'------'])
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 
        
            #load best model
            best_state = safe_torch_load(os.path.join(exp_dir(opt, fold), 'best.pth'))
            model.load_state_dict(best_state['state_dict'])
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True,
                persistent_workers=opt.n_threads > 0)
                
            #---------
            
            test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'both', '------test_val_modality_dist:','both','------'])
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
            test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                            logger=test_logger,modality='both',dist=opt.test_val_modality_dist)
            
            with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                    f.write('test set:'+'modality: '+opt.test_val_modality+';dist: '+opt.test_val_modality_dist+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1_m) 

            # ---- 追加一行结构化结果，供 plot_all.py 读取作图（问题1/问题2 用）----
            if getattr(opt, 'summary_csv', ''):
                _append_summary(opt, fold, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v)

            # ==== 以下 6 个「模态破坏」组合默认执行；
            # 加 --test_only_both 可只跑上面那组干净测试，省掉 6/7 的测试时间 ====
            if not getattr(opt, 'test_only_both', False):
                #---------
                test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'audio', '------test_val_modality_dist:','noise','------'])
                test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                                logger=test_logger,modality='audio',dist='noise')
            
                with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                        f.write('test set:'+'modality: '+'audio'+';dist: '+'noise'+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
                 #---------
                test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'audio', '------test_val_modality_dist:','addnoise','------'])
                test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                                logger=test_logger,modality='audio',dist='addnoise')
            
                with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                        f.write('test set:'+'modality: '+'audio'+';dist: '+'addnoise'+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
                #---------
                test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'audio', '------test_val_modality_dist:','zeros','------'])
                test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                                logger=test_logger,modality='audio',dist='zeros')
            
                with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                        f.write('test set:'+'modality: '+'audio'+';dist: '+'zeros'+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
                #---------
                test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'video', '------test_val_modality_dist:','noise','------'])
                test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                                logger=test_logger,modality='video',dist='noise')
            
                with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                        f.write('test set:'+'modality: '+'video'+';dist: '+'noise'+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
                #---------
                test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'video', '------test_val_modality_dist:','addnoise','------'])
                test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                                logger=test_logger,modality='video',dist='addnoise')
            
                with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                        f.write('test set:'+'modality: '+'video'+';dist: '+'addnoise'+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
                #---------
                test_logger.logger.writerow(['------fold:',fold,'------test_val_modality:', 'video', '------test_val_modality_dist:','zeros','------'])
                test_loss, test_prec1_m, test_prec1_a, test_prec1_v, test_bacc_m, test_bacc_a, test_bacc_v = val_epoch(epoch=10000, data_loader=test_loader, model=model, criterion=criterion, opt=opt,
                                                logger=test_logger,modality='video',dist='zeros')
            
                with open(os.path.join(exp_dir(opt, fold), 'test_set_bestval.txt'), 'a') as f:
                        f.write('test set:'+'modality: '+'video'+';dist: '+'zeros'+'Prec1_m: ' + str(test_prec1_m) + '; Prec1_a: ' + str(test_prec1_a) +'Prec1_v: ' + str(test_prec1_v) +'; Loss: ' + str(test_loss))
                    
            
    # 各折均值±标准差写在实验根目录，不再混进 result_path 的公共文件
    with open(os.path.join(exp_dir(opt), 'summary.txt'), 'a') as f:
        f.write('%s  folds=%s  M=%.4f+-%.4f\n' % (
            opt.store_name, folds_to_run,
            float(np.mean(np.array(test_accuracies))) if test_accuracies else float('nan'),
            float(np.std(np.array(test_accuracies))) if test_accuracies else float('nan')))
    print('==> %s  M = %s' % (opt.store_name,
                              '%.2f+-%.2f' % (np.mean(test_accuracies), np.std(test_accuracies))
                              if test_accuracies else 'n/a'))
