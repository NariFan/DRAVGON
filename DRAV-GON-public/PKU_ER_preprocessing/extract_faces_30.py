# -*- coding: utf-8 -*-
"""
30 帧版人脸提取（对照 extract_faces.py 的 15 帧）。

动机：论文 3.3.1 节明确写 "Each input video is decomposed into 30 frames"，
而仓库默认 extract_faces.py 用的是 save_frames=15。时序采样点翻倍，
为情绪识别提供更丰富的时间动态信息，期望提升 K 折性能。

输出到独立目录 Video_processed_face_30f，与 15 帧版共存、互不影响。
"""
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('device:', device)

mtcnn = MTCNN(
    image_size=(720, 1280),
    margin=40,
    thresholds=[0.6, 0.7, 0.7],
    device=device
).to(device)

save_frames = 30          # 与论文一致：30 帧
input_fps = 25
save_length = 3.6

root = os.environ.get('PKU_ER_ROOT', 'data/PKU-AV')
raw_dir = os.path.join(root, 'Raw')
processed_dir = os.path.join(root, 'Processed/Video_processed_face_30f')

select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(int(save_length * input_fps), total_frames)
    start_frame = max(0, (total_frames - target_frames) // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    numpy_video = []
    selected_frames = select_distributed(save_frames, target_frames)
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret or current_frame >= target_frames:
            break
        if current_frame in selected_frames:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    face = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
                    numpy_video.append(face)
                else:
                    numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
                selected_frames.remove(current_frame)
            except Exception as e:
                numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
        current_frame += 1
    while len(numpy_video) < save_frames:
        numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return np.array(numpy_video)


for video_folder in tqdm(os.listdir(raw_dir)):
    video_dir = os.path.join(raw_dir, video_folder)
    if not os.path.isdir(video_dir):
        continue
    output_folder = os.path.join(processed_dir, video_folder)
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(video_dir):
        if not filename.endswith('.mp4'):
            continue
        input_path = os.path.join(video_dir, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_facecroppad.npy")
        if os.path.exists(output_path):
            continue
        try:
            np.save(output_path, process_video(input_path))
        except Exception as e:
            print(f"Failed {filename}: {e}")

print("All videos processed (30 frames)!")
