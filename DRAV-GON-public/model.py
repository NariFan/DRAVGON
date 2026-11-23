'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

from torch import nn

from models import multimodalcnn

def generate_model(opt):
    assert opt.model in ['multimodalcnn']

    if opt.model == 'multimodalcnn':
        model = multimodalcnn.MultiModalCNN(
            opt.n_classes,
            fusion=opt.fusion,
            seq_length=opt.sample_duration,
            pretr_ef=opt.pretrain_path,
            num_heads=opt.num_heads,
            har=getattr(opt, 'har', 'full'),
            cross_dir=getattr(opt, 'cross_dir', 'both'),
            cross_depth=getattr(opt, 'cross_depth', 1),
        )

    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        # 消融实验要在论文里报「交叉注意力参数量」这一列（单向 vs 双向的容量匹配依据），
        # DataParallel 包了一层，要取 .module。
        inner = model.module if hasattr(model, 'module') else model
        if hasattr(inner, 'cross_params'):
            print("Cross-attention parameters: ", inner.cross_params(),
                  "(av blocks: %d, va blocks: %d)" % (len(inner.av_blocks), len(inner.va_blocks)))
        
    
    return model, model.parameters()
