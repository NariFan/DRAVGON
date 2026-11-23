# -*- coding: utf-8 -*-

import librosa
import os
import soundfile as sf
import numpy as np

# 设置PKU-AV数据集根目录
root = os.environ.get('PKU_ER_ROOT', 'data/PKU-AV')
raw_dir = os.path.join(root, 'Raw')  # 原始数据目录
processed_dir = os.path.join(root, 'Processed/Audio_processed')  # 处理后的输出目录
target_time = 3.6  # 目标时长（秒）

# 创建输出目录（如果不存在）
os.makedirs(processed_dir, exist_ok=True)

# 遍历所有video目录
for video_folder in os.listdir(raw_dir):
    video_path = os.path.join(raw_dir, video_folder)

    # 跳过非目录文件
    if not os.path.isdir(video_path):
        continue

    # 遍历每个video目录中的mp4文件
    for filename in os.listdir(video_path):
        if not filename.endswith('.mp4'):
            continue

        input_path = os.path.join(video_path, filename)

        try:
            # 加载音频（自动重采样到22050Hz）
            y, sr = librosa.load(input_path, sr=22050)

            # 预处理逻辑保持不变
            target_length = int(sr * target_time)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            else:
                remain = len(y) - target_length
                y = y[remain // 2: -(remain - remain // 2) or None]

            # 创建输出目录结构
            output_folder = os.path.join(processed_dir, video_folder)
            os.makedirs(output_folder, exist_ok=True)

            # 生成输出文件名（改为wav格式）
            output_name = f"{os.path.splitext(filename)[0]}_croppad.wav"
            output_path = os.path.join(output_folder, output_name)

            # 保存处理后的音频
            sf.write(output_path, y, sr)
            print(f"Processed: {input_path} -> {output_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {repr(e)}")

print("All files processed!")