# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print('device:', device)

# 初始化MTCNN参数
mtcnn = MTCNN(
    image_size=(720, 1280),
    margin=40,
    thresholds=[0.6, 0.7, 0.7],
    device=device
).to(device)

# 视频处理参数
save_frames = 15  # 需要保存的帧数
input_fps = 25  # 输入视频的帧率
save_length = 3.6  # 需要处理的视频长度（秒）
save_avi = False  # 是否保存AVI文件

# 数据集路径配置
root = os.environ.get('PKU_ER_ROOT', 'data/PKU-AV')
raw_dir = os.path.join(root, 'Raw')  # 原始视频目录
processed_dir = os.path.join(root, 'Processed/Video_processed_face')  # 处理结果目录

# 辅助函数：均匀采样帧
select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]


def process_video(video_path, output_dir):
    """处理单个视频的核心函数"""        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算实际需要处理的帧数
    target_frames = min(int(save_length * input_fps), total_frames)
    start_frame = max(0, (total_frames - target_frames) // 2)

    # 重新设置视频读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 初始化存储容器
    numpy_video = []
    selected_frames = select_distributed(save_frames, target_frames)
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret or current_frame >= target_frames:
            break

        if current_frame in selected_frames:
            try:
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 人脸检测
                boxes, _ = mtcnn.detect(frame_rgb)

                if boxes is not None and len(boxes) > 0:
                    # 取最大的人脸框
                    box = boxes[0].astype(int)
                    x1, y1, x2, y2 = box
                    face = frame[y1:y2, x1:x2]

                    # 调整尺寸并保存
                    face = cv2.resize(face, (224, 224))
                    numpy_video.append(face)
                else:
                    numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

                selected_frames.remove(current_frame)

            except Exception as e:
                print(f"Error processing frame {current_frame}: {str(e)}")
                numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

        current_frame += 1

    # 填充缺失帧
    while len(numpy_video) < save_frames:
        numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return np.array(numpy_video)


# 主处理流程
for video_folder in tqdm(os.listdir(raw_dir)):
    video_dir = os.path.join(raw_dir, video_folder)

    # 跳过非目录文件
    if not os.path.isdir(video_dir):
        continue

    # 创建输出目录
    output_folder = os.path.join(processed_dir, video_folder)
    os.makedirs(output_folder, exist_ok=True)

    # 处理目录中的每个视频
    for filename in os.listdir(video_dir):
        if not filename.endswith('.mp4'):
            continue

        input_path = os.path.join(video_dir, filename)
        output_path = os.path.join(
            output_folder,
            f"{os.path.splitext(filename)[0]}_facecroppad.npy"
        )

        # 跳过已处理文件
        if os.path.exists(output_path):
            continue

        try:
            processed_video = process_video(input_path, output_folder)
            np.save(output_path, processed_video)
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")

print("All videos processed!")