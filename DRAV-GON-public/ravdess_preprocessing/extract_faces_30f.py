# -*- coding: utf-8 -*-
"""
RAVDESS 30 帧版人脸提取（对照 extract_faces.py 的 15 帧）。

动机：论文 3.3.1 写 "Each input video is decomposed into 30 frames"，
15 帧会丢掉一半时序信息。PKU-ER 上已实测 30 帧比 15 帧高 0.98~4.40 个点。

只提取去重后标注里实际用到的视频（modality=01 的 1425 条），
不处理纯视频(02)与重复项，省掉一半以上的无用计算。

输出：Video_Speech_Actors_01-24_processed_30f/<Actor>/<name>_facecroppad.npy
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('device:', device)

mtcnn = MTCNN(image_size=(720, 1280), margin=40,
              thresholds=[0.6, 0.7, 0.7], device=device).to(device)

save_frames = 30
input_fps = 30
save_length = 3.6

root = os.environ.get('RAVDESS_ROOT', 'data/RAVDESS')
out_root = os.path.join(root, 'Video_Speech_Actors_01-24_processed_30f')
annot = 'ravdess_preprocessing/annotations_00.txt'

select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]


def npy_to_mp4(video_path):
    """
    标注里的「视频路径」其实是**已处理的 npy 文件**（不是原始 mp4）。
    直接拿它给 VideoCapture 打开会立即失败，循环一帧都读不到，
    结果整段被零填充成全黑（曾让视觉头退化到随机猜测的 13%）。
    这里反推回原始 mp4 路径：
      .../Video_Speech_Actors_01-24_processed[_30f]/Actor_01/xx_facecroppad.npy
      -> .../Video_Speech_Actors_01-24/Actor_01/xx.mp4
    """
    p = video_path.replace('_processed_30f', '').replace('_processed', '')
    p = p.replace('_facecroppad.npy', '.mp4')
    # 目录名去掉 _processed 后缀
    p = p.replace('/Video_Speech_Actors_01-24/', '/Video_Speech_Actors_01-24/')
    return p


def video_to_out(video_path):
    """标注里的 npy 路径 -> 30 帧输出 npy 路径（同目录名换成 _30f）"""
    actor = video_path.split('/')[-2]
    name = os.path.basename(video_path)
    return os.path.join(out_root, actor, name)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    framen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = int(save_length * input_fps)
    if target > framen:
        target = framen
    # 注意：不能用 cap.set(CAP_PROP_POS_FRAMES) 跳帧——在 RAVDESS 的 mp4 上该定位会失效，
    # 导致读到空帧、整段被填成全黑（曾让视觉头退化到随机猜测的 13%）。
    # 改用与原版 extract_faces.py 一致的做法：顺序读取并丢弃开头若干帧。
    for _ in range(max(0, framen - target) // 2):
        cap.read()

    frames_to_select = set(select_distributed(save_frames, target))
    out = []
    idx = 0
    while True:
        ret, im = cap.read()
        if not ret or idx >= target:
            break
        if idx in frames_to_select:
            try:
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(im_rgb)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    face = cv2.resize(im[y1:y2, x1:x2], (224, 224))
                    out.append(face)
                else:
                    out.append(np.zeros((224, 224, 3), dtype=np.uint8))
            except Exception:
                out.append(np.zeros((224, 224, 3), dtype=np.uint8))
        idx += 1
    while len(out) < save_frames:
        out.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return np.array(out)


# 从去重后的标注取视频清单
videos = []
with open(annot) as f:
    for line in f:
        line = line.strip()
        if line:
            videos.append(line.split(';')[0])

print('待提取视频:', len(videos))
done = 0
for vp in tqdm(videos):
    op = video_to_out(vp)
    if os.path.exists(op):
        continue
    src = npy_to_mp4(vp)
    if not os.path.exists(src):
        print('缺失:', src)
        continue
    os.makedirs(os.path.dirname(op), exist_ok=True)
    try:
        np.save(op, process_video(src))
        done += 1
    except Exception as e:
        print('失败', vp, e)

print('30 帧提取完成，新算 %d 条 -> %s' % (done, out_root))
