# -*- coding: utf-8 -*-
import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('device:',device)

mtcnn = MTCNN(image_size=(720, 1280), device=device)

mtcnn.to(device)
print('mtcnn.device',mtcnn.device)
save_frames = 15
input_fps = 30
save_length = 3.6 #seconds
save_avi = False

failed_videos = []

root = os.environ.get('RAVDESS_ROOT', 'data/RAVDESS')
files = 'Video_Speech_Actors_01-24'
videofiles = os.path.join(root,files)


select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
n_processed = 0
for sess in tqdm(sorted(os.listdir(videofiles))):   
    for filename in os.listdir(os.path.join(videofiles, sess)):
           
        if filename.endswith('.mp4'):
            # 打开视频文件  
            cap = cv2.VideoCapture(os.path.join(videofiles, sess, filename))  
            # 计算视频的总帧数 
            framen = 0
            while True:
                i,q = cap.read()
                if not i:
                    break
                framen += 1
            # 重新打开视频文件，以便重新读取 
            cap = cv2.VideoCapture(os.path.join(videofiles, sess, filename))
            # 如果视频长度不足以保存指定秒数的帧，则计算跳过的开始帧数 
            if save_length*input_fps > framen:                    
                skip_begin = int((framen - (save_length*input_fps)) // 2)
                for i in range(skip_begin):
                    _, im = cap.read() 
                    
            framen = int(save_length*input_fps) # 设置要处理的帧数     
            frames_to_select = select_distributed(save_frames,framen)# 从总帧数中均匀选择指定数量的帧  
            save_fps = save_frames // (framen // input_fps) 

            # 如果需要保存为AVI文件，则初始化VideoWriter
            if save_avi:
                out = cv2.VideoWriter(os.path.join(videofiles,'Video_Speech_Actors_01-24_processed', sess, filename[:-4]+'_facecroppad.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), save_fps, (224,224))

            numpy_video = []
            success = 0
            frame_ctr = 0
            # 遍历视频中的每一帧 
            while True: 
                ret, im = cap.read()
                if not ret:
                    break
                # 如果当前帧不在选择的帧列表中，则跳过  
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    # 如果处理失败，记录视频信息
                    failed_videos.append((sess, i))
                    break

                # 交换BGR图像的红色和蓝色通道，以匹配MTCNN的输入要求
                temp = im[:,:,-1]
                im_rgb = im.copy()
                im_rgb[:,:,-1] = im_rgb[:,:,0]
                im_rgb[:,:,0] = temp
                # 将图像转换为Tensor并移动到指定设备 
                im_rgb = torch.tensor(im_rgb)
                im_rgb = im_rgb.to(device)

                bbox = mtcnn.detect(im_rgb)
                if bbox[0] is not None:
                    bbox = bbox[0][0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                im = im[y1:y2, x1:x2, :]
                im = cv2.resize(im, (224,224))
                if save_avi:
                    out.write(im)
                numpy_video.append(im)
            if len(frames_to_select) > 0:
                for i in range(len(frames_to_select)):
                    if save_avi:
                        out.write(np.zeros((224,224,3), dtype = np.uint8))
                    numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
            if save_avi:
                out.release() 

            outfile = os.path.join(root,'Video_Speech_Actors_01-24_processed', sess)
            # 如果目录不存在，则创建它  
            if not os.path.exists(outfile):  
                os.makedirs(outfile)
            # 保存处理后的音频文件，文件名添加'_croppad'后缀  
            np.save(os.path.join(outfile,filename[:-4]+'_facecroppad.npy'), np.array(numpy_video))
            if len(numpy_video) != 15:
                print('Error', sess, filename)    
                            
    n_processed += 1      
    with open('processed.txt', 'a') as f:
        f.write(sess + '\n')
    print(failed_videos)
