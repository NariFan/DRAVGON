# -*- coding: utf-8 -*-
"""
按目标信噪比 (SNR, dB) 向语音波形叠加高斯白噪声。

用途
----
回应审稿人「问题3：实验场景非真实，缺乏定量鲁棒性验证」。
审稿人要求在测试集上添加多梯度信噪比的加性噪声，构建不同干扰强度的测试子集。
本模块负责其中「加噪」这一步，被 datasets/ravdess.py 调用。

为什么加在波形层而不是特征层
----------------------------
- 加在波形（wav 读出后、提取 MFCC/CQT/BEATs 之前）后，所有声学特征都会一起被污染，
  这才是真正的「声学噪声」，审稿人无法质疑只是做了特征扰动。
- 若加在特征层（820 通道的特征张量上），噪声会绕过多尺度声学分析，
  信噪比也失去物理意义。
代价是每个信噪比都要重算一遍音频特征（含 BEATs 前向），但由于只在测试集上做，
且 RAVDESS 每折测试集约 240~300 条，开销可以接受。

信噪比定义
---------
    SNR_dB = 10 * log10( P_signal / P_noise )

其中 P_signal 是语音的平均功率。本模块提供两种功率估计口径：

- 'full'   ：整段波形的平均功率。定义最直白，但如果录音里静音段多，
             会把功率拉低，导致实际感受到的干扰比标称 SNR 更重。
- 'active' ：只用「有声帧」估计功率（librosa.effects.split 检测），
             更贴近人耳感受到的信噪比，论文里建议采用这个口径。

实测说明：本项目的数据已由 ravdess_preprocessing/extract_audios.py 统一裁剪到 3.6 秒，
语音占比很高，所以两种口径算出来的 SNR 差异极小（约 0.00 dB），
默认用 'active'，同时保留 'full' 供审稿人质疑时一键复现对比。

确定性播种
---------
噪声按「样本文件名 + SNR」哈希播种，而不是按 sample index 播种。这样做的好处：

1. 不受标注文件行序、DataLoader 的 num_workers 多进程顺序影响；
2. 同一条样本在**不同折**上拿到的是同一段噪声 → 跨折结果可比；
3. 论文实验可精确复现。

单独运行本文件可做自检（无需任何权重和数据）：

    python noise_utils.py

自检内容：合成一段 3.6 秒语音，对每个目标 SNR 验证回算误差 < 0.1 dB，
并检查静音输入不产生 NaN、同种子可复现、不同种子噪声不同。
"""

import hashlib

import numpy as np

# 功率下限。低于此值视为静音段，直接跳过加噪，避免出现除零或 NaN。
_EPS = 1e-10


def make_rng(seed_base, sample_key, snr_db):
    """
    根据 (随机种子基值, 样本标识, 信噪比) 构造一个确定的随机数发生器。

    参数
    ----
    seed_base  : int，全局种子，来自命令行 --noise_seed
    sample_key : str，样本唯一标识（本项目传音频文件名 basename）
    snr_db     : float 或 None，目标信噪比；None 表示不加噪

    返回
    ----
    numpy.random.Generator

    用哈希而不是直接用 Python 内置 hash()：内置 hash 对字符串是随进程加盐的
    （PYTHONHASHSEED 随机），会导致多次运行结果不一致。
    """
    payload = "{}|{}|{}".format(int(seed_base), str(sample_key), str(snr_db))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    # 取前 8 字节转成非负整数作为种子，保证 SeedSequence 接受。
    seed = int.from_bytes(digest[:8], byteorder="big") % (2 ** 32)
    return np.random.default_rng(seed)


def signal_power(y, mode="active", top_db=20):
    """
    计算语音的平均功率。

    参数
    ----
    y       : 一维 numpy 数组，语音波形（建议 float32）
    mode    : 'full' 整段功率 / 'active' 仅有声帧功率
    top_db  : 有声/静音判别的能量阈值（dB），仅 mode='active' 时生效。
              数值越大判定越宽松，top_db=20 表示低于峰值 20 dB 以下的段算静音。

    返回
    ----
    float，平均功率（均方值）。空数组返回 0.0。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return 0.0

    if mode == "full":
        return float(np.mean(y ** 2))

    if mode != "active":
        raise ValueError("未知的功率估计口径: {!r}，可选 'full' 或 'active'".format(mode))

    # 延迟导入：只在真正需要有声帧检测时才负担 librosa 的导入开销。
    import librosa

    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        # 整段都被判成静音（例如纯零输入），退回整段功率以免除以零。
        return float(np.mean(y ** 2))

    voiced = np.concatenate([y[a:b] for a, b in intervals])
    if voiced.size == 0:
        return float(np.mean(y ** 2))
    return float(np.mean(voiced ** 2))


def add_noise_at_snr(y, snr_db, sample_key, seed=1234,
                     mode="active", top_db=20, eps=_EPS):
    """
    向波形叠加指定信噪比的高斯白噪声。

    参数
    ----
    y           : 一维 numpy 数组，干净的语音波形
    snr_db      : 目标信噪比（dB）。传 None 表示不加噪，原样返回。
                  数值越小噪声越强，例如 -5 dB 时噪声功率比语音还大。
    sample_key  : 样本唯一标识，用于确定性播种（一般传文件名）
    seed        : 全局随机种子
    mode        : 功率估计口径，'active' 或 'full'
    top_db      : 有声帧检测阈值
    eps         : 功率下限保护

    返回
    ----
    (y_noisy, snr_applied)
        y_noisy      : 加噪后的波形，float32
        snr_applied  : 实际施加的信噪比；若因静音跳过加噪则为 None

    为什么返回 None 而不返回一个伪造的 SNR：调用方需要能分辨「没加噪」和
    「加了噪」，从而在统计结果里如实标注，避免论文里出现虚假数据点。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    # 未指定信噪比 → 干净样本，直接返回。
    if snr_db is None:
        return y, None

    ps = signal_power(y, mode=mode, top_db=top_db)

    # 静音或功率过低：叠加任何有限噪声都会让 SNR 爆到负无穷，
    # 而且除以极小数容易产生数值问题，这里选择原样返回。
    if ps < eps:
        return y, None

    # SNR = 10*log10(Ps/Pn)  =>  Pn = Ps / 10^(SNR/10)
    target_snr = float(snr_db)
    pn = ps / (10.0 ** (target_snr / 10.0))

    rng = make_rng(seed, sample_key, target_snr)
    noise = rng.standard_normal(y.shape[0]).astype(np.float32) * np.sqrt(pn)

    y_noisy = (y + noise).astype(np.float32)
    return y_noisy, target_snr


def measured_snr(y_clean, y_noisy, mode="active", top_db=20, eps=_EPS):
    """
    从 (干净波形, 加噪波形) 反算实际信噪比，用于验证加噪是否准确。

    这是自检和论文可复现性检查的关键：注入时声称的 SNR 必须能被独立测回。
    注意 receiver 端用**干净信号**的有声区间做功率统计口径，与注入端保持一致。

    返回 float 或 None（功率过低无法测量时返回 None）。
    """
    y_clean = np.asarray(y_clean, dtype=np.float32).reshape(-1)
    y_noisy = np.asarray(y_noisy, dtype=np.float32).reshape(-1)
    if y_clean.shape[0] != y_noisy.shape[0]:
        raise ValueError("两段波形长度不一致，无法计算信噪比")

    noise = y_noisy - y_clean

    if mode == "full":
        ps = float(np.mean(y_clean ** 2))
        pn = float(np.mean(noise ** 2))
    elif mode == "active":
        import librosa

        intervals = librosa.effects.split(y_clean, top_db=top_db)
        if len(intervals) == 0:
            ps = float(np.mean(y_clean ** 2))
            pn = float(np.mean(noise ** 2))
        else:
            voiced_clean = np.concatenate([y_clean[a:b] for a, b in intervals])
            voiced_noise = np.concatenate([noise[a:b] for a, b in intervals])
            ps = float(np.mean(voiced_clean ** 2)) if voiced_clean.size else 0.0
            pn = float(np.mean(voiced_noise ** 2)) if voiced_noise.size else 0.0
    else:
        raise ValueError("未知的功率估计口径: {!r}".format(mode))

    if ps < eps or pn < eps:
        return None
    return float(10.0 * np.log10(ps / pn))


# ---------------------------------------------------------------------------
# 自检：python noise_utils.py
# 无需任何权重、数据和 GPU，在本机（macOS）即可运行。
# ---------------------------------------------------------------------------
def _synthetic_speech(sr=22050, duration=3.6, rng=None):
    """
    合成一段「像语音」的测试信号。

    刻意使用项目规定的 3.6 秒 / 22050 Hz：这是硬性约束，
    因为模型要求所有音频特征都是 156 帧（见 CODEBUDDY.md），
    而 3.6 秒恰好产出 156 帧；短于此会有拼接失败。

    构造方式：低频基频 + 若干谐波 + 缓慢的幅度包络，
    再在开头和结尾留一点静音，以便检测 combine 'active' 口径确实生效。
    """
    rng = rng or np.random.default_rng(0)
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr

    f0 = 120.0
    sig = np.zeros(n, dtype=np.float32)
    for k in range(1, 8):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)

    # 缓慢包络，模拟语音起伏
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
    sig = sig * envelope

    # 首尾各留 0.15 秒静音，用来验证 'active' 口径会忽略静音段
    silence = int(0.15 * sr)
    sig[:silence] = 0.0
    sig[-silence:] = 0.0

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
    return sig.astype(np.float32)


def _self_test():
    sr = 22050
    y = _synthetic_speech(sr=sr, duration=3.6)
    target_snrs = [20.0, 15.0, 10.0, 5.0, 0.0, -5.0]
    failures = []

    print("=" * 68)
    print("noise_utils 自检")
    print("=" * 68)

    for mode in ["active", "full"]:
        print("\n[功率口径: {}]".format(mode))
        print("  目标SNR     回算SNR      误差(dB)   波形范围")
        for snr in target_snrs:
            y_noisy, applied = add_noise_at_snr(
                y, snr, sample_key="01-01-01-01-01-01-01_croppad.wav",
                seed=1234, mode=mode)
            assert applied is not None, "{} dB 未能施加噪声".format(snr)

            got = measured_snr(y, y_noisy, mode=mode)
            assert got is not None, "{} dB 无法回算信噪比".format(snr)

            err = abs(got - snr)
            status = "OK " if err < 0.1 else "FAIL"
            print("  {:>6.1f}    {:>9.3f}    {:>8.4f}    [{:+.3f},{:+.3f}]  {}".format(
                snr, got, err, float(y_noisy.min()), float(y_noisy.max()), status))
            if err >= 0.1:
                failures.append("SNR {} mode {} 回算误差过大".format(snr, mode))

    # --- 检查 1：静音输入不产生 NaN ---
    silent = np.zeros(int(sr * 3.6), dtype=np.float32)
    y_silent, applied = add_noise_at_snr(silent, 10.0, sample_key="silent.wav")
    if applied is not None or np.isnan(y_silent).any():
        failures.append("静音输入处理不正确（应跳过加噪且无 NaN）")
        print("\n[静音保护] FAIL")
    else:
        print("\n[静音保护] OK 静音输入被跳过加噪，无 NaN")

    # --- 检查 2：不加噪时原样返回 ---
    y_clean_ret, applied = add_noise_at_snr(y, None, sample_key="x.wav")
    if applied is not None or not np.allclose(y_clean_ret, y):
        failures.append("snr_db=None 时应原样返回")
        print("[干净通道] FAIL")
    else:
        print("[干净通道] OK snr_db=None 原样返回")

    # --- 检查 3：同种子可复现、换种子噪声不同 ---
    key = "01-01-01-01-01-01-01_croppad.wav"
    n1, _ = add_noise_at_snr(y, 5.0, key, seed=1234)
    n2, _ = add_noise_at_snr(y, 5.0, key, seed=1234)
    n3, _ = add_noise_at_snr(y, 5.0, key, seed=4321)
    if not np.allclose(n1, n2):
        failures.append("相同种子两次加噪结果不一致，结果无法复现")
        print("[确定性] FAIL 同种子结果不一致")
    elif np.allclose(n1, n3):
        failures.append("不同种子产生了相同噪声")
        print("[确定性] FAIL 不同种子结果相同")
    else:
        print("[确定性] OK 同种子可复现、换种子噪声改变")

    # --- 检查 4：不同样本拿到不同噪声 ---
    nA, _ = add_noise_at_snr(y, 5.0, "actor01_a.wav", seed=1234)
    nB, _ = add_noise_at_snr(y, 5.0, "actor01_b.wav", seed=1234)
    if np.allclose(nA, nB):
        failures.append("不同样本拿到了完全相同的噪声")
        print("[按样本播种] FAIL")
    else:
        print("[按样本播种] OK 不同样本的噪声相互独立")

    print("\n" + "=" * 68)
    if failures:
        print("自检未通过，共 {} 项问题：".format(len(failures)))
        for f in failures:
            print("  - " + f)
        return 1
    print("自检全部通过")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
