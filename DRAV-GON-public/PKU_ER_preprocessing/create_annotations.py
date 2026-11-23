import os
import pandas as pd
from chardet import detect
# 配置路径
dataset_root = os.environ.get('PKU_ER_ROOT', 'data/PKU-AV')
video_processed_dir = os.path.join(dataset_root, "Processed/Video_processed_face")
audio_processed_dir = os.path.join(dataset_root, "Processed/Audio_processed")
label_csv_path = os.path.join(dataset_root, "label.csv")
output_txt_path = os.path.join(dataset_root, "dataset_split.txt")

# 自动检测编码
with open(label_csv_path, 'rb') as f:
    rawdata = f.read()
    encoding = detect(rawdata)['encoding']

# 处理特殊编码情况,加载标签数据
try:
    print('encoding',encoding)
    df = pd.read_csv(label_csv_path, encoding=encoding)
    print('encoding',encoding)
except UnicodeDecodeError:
    df = pd.read_csv(label_csv_path, encoding='gb18030')


# 创建映射字典（根据实际需求调整）
split_mapping = {
    "train": "training",
    "valid": "validation",
    "test": "testing"
}

# 生成条目列表
entries = []
for _, row in df.iterrows():
    # 构建视频文件路径
    video_path = os.path.join(
        video_processed_dir,
        row["video_id"],
        f"{row['clip_id']}_facecroppad.npy"
    )

    # 构建音频文件路径（假设音频目录结构与视频一致）
    audio_path = os.path.join(
        audio_processed_dir,
        row["video_id"],
        f"{row['clip_id']}_croppad.wav"
    )

    # 验证文件存在性
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found - {video_path}")
        continue

    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found - {audio_path}")
        continue

    # 映射数据集划分
    dataset_split = split_mapping.get(row["mode"].lower(), "unknown")

    # 标签映射（关键，写错会让训练直接崩）
    # label.csv 里 label 是情感极性 {-1:负, 0:中, 1:正}，
    # 而 datasets/ravdess.py::make_dataset 会对标注文件里的标签再做一次 int(label)-1，
    # 直接写 -1/0/1 会得到 -2/-1/0 这些负标签，CrossEntropyLoss 立刻报错。
    # 本映射把极性转成 1/2/3，减 1 之后正好是 0/1/2，与 --n_classes 3 对应，
    # 且与仓库里既有的 PKU_ER_preprocessing/annotations.txt 完全一致（三档数量 288/291/323）。
    label_mapping = {-1: 2, 0: 3, 1: 1}
    label = label_mapping[row["label"]]

    # 构建条目
    entry = f"{video_path};{audio_path};{label};{dataset_split}"
    entries.append(entry)

# 写入txt文件
with open(output_txt_path, "w") as f:
    f.write("\n".join(entries))

print(f"Successfully generated {len(entries)} entries in {output_txt_path}")
print(f"Failed to process {len(df) - len(entries)} entries")