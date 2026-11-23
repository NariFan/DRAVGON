# -*- coding: utf-8 -*-
"""
把按 Actor 平铺的原始 mp4 整理成预处理脚本期望的目录结构。

本机下载到的原始数据：
    ROOT/Actor_01/01-01-01-01-01-01-01.mp4   (modality 01 = 音视频)
    ROOT/Actor_01/02-01-01-01-01-01-01.mp4   (modality 02 = 纯视频)
    每个 Actor 120 条，共 2880 条

预处理脚本期望的官方 zip 结构：
    ROOT/Video_Speech_Actors_01-24/Actor_01/01-01-01-01-01-01-01.mp4
    ROOT/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav

只用 modality=01 且 vocal=01（speech）的 1440 条。

本脚本做两件事：
  1. 视频：1440 条 01-01-*.mp4 软链到 Video_Speech_Actors_01-24/（不复制，省 6.5G）
  2. 音频：从对应 mp4 抽音轨写成 03-*.wav 到 Audio_Speech_Actors_01-24/

音频必须抽：官方数据集另有 Audio_Speech_Actors_*.zip，但 download_ravdess.py
只下了视频 zip，原始 wav 不存在，只能从 mp4 音轨提取。
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg

ROOT = os.environ.get('RAVDESS_ROOT', 'data/RAVDESS')
SRC = ROOT
VID_OUT = os.path.join(ROOT, 'Video_Speech_Actors_01-24')
AUD_OUT = os.path.join(ROOT, 'Audio_Speech_Actors_01-24')

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ACTORS = ['Actor_%02d' % i for i in range(1, 25)]


def collect():
    items = []
    for actor in ACTORS:
        d = os.path.join(SRC, actor)
        if not os.path.isdir(d):
            print('[warn] 缺目录', d)
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith('.mp4'):
                continue
            if not name.startswith('01-01-'):
                continue
            items.append((actor, name))
    return items


def link_video(actor, name):
    src = os.path.join(SRC, actor, name)
    dst = os.path.join(VID_OUT, actor, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    return dst


def extract_audio(actor, name):
    src = os.path.join(SRC, actor, name)
    wav_name = '03' + name[2:-4] + '.wav'
    dst = os.path.join(AUD_OUT, actor, wav_name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
           '-i', src, '-vn', '-acodec', 'pcm_s16le', dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('%s : %s' % (src, r.stderr.strip()[:300]))
    return dst


def main():
    items = collect()
    print('speech 样本数:', len(items))
    assert len(items) == 1440, '期望 1440 条，实际 %d' % len(items)

    for actor, name in items:
        link_video(actor, name)
    print('视频软链完成', VID_OUT)

    done = 0
    errs = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(extract_audio, a, n): (a, n) for a, n in items}
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as e:
                errs.append(str(e))
            if done % 200 == 0:
                print('  已抽音频 %d / %d' % (done, len(items)), flush=True)
    print('音频抽取完成: %d 成功, %d 失败' % (done, len(errs)))
    print('输出目录:', AUD_OUT)
    for e in errs[:10]:
        print('  [ERR]', e)


if __name__ == '__main__':
    main()
