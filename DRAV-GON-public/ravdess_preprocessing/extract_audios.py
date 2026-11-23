# -*- coding: utf-8 -*-

import librosa
import os
import soundfile as sf
import numpy as np

#audiofile = 'E://OpenDR_datasets//RAVDESS//Actor_19//03-01-07-02-01-02-19.wav'
##this file preprocess audio files to ensure they are of the same length. if length is less than 3.6 seconds, it is padded with zeros in the end. otherwise, it is equally cropped from 
##both sides


# 设置音频文件的根目录（请确保此路径正确）

root = os.environ.get('RAVDESS_ROOT', 'data/RAVDESS')
files = 'Audio_Speech_Actors_01-24'
audiofiles = os.path.join(root,files)
target_time = 3.6 #sec
# 确保当前遍历的是一个文件夹（演员文件夹）  
for actor in os.listdir(audiofiles):
     # 遍历演员文件夹下的每个音频文件  
    for audiofile in os.listdir(os.path.join(audiofiles, actor)):
        # 检查文件是否为.wav格式且文件名中不包含'croppad'
        if not audiofile.endswith('.wav') or 'croppad' in audiofile:
            continue

        audios = librosa.core.load(os.path.join(audiofiles, actor, audiofile), sr=22050)
	
        y = audios[0]
        sr = audios[1]
        target_length = int(sr * target_time)# 计算目标长度（以样本点为单位）
        # 判断音频长度并进行裁剪或填充  
        if len(y) < target_length:
            y = np.array(list(y) + [0 for i in range(target_length - len(y))])
        else:
            remain = len(y) - target_length
            y = y[remain//2:-(remain - remain//2)]

        outfile = os.path.join(root,'Audio_Speech_Actors_01-24_processed', actor)
        # 如果目录不存在，则创建它  
        if not os.path.exists(outfile):  
            os.makedirs(outfile)
	    # 保存处理后的音频文件，文件名添加'_croppad'后缀  
        sf.write(os.path.join(outfile,audiofile[:-4]+'_croppad.wav'), y, sr)

	
