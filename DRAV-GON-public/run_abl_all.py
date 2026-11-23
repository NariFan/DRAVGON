# -*- coding: utf-8 -*-
"""
RAVDESS 消融实验总控脚本（无人值守，可断点续跑）。

做什么
------
按「优先级顺序」把 run_abl_ravdess.py 里的每一格的每一折依次跑完：

    Panel 1（G²AF × HAR 因子设计） -> Panel 2（DIP） -> Panel 3（单向/双向） -> Panel 4（补充）

万一时间不够被中断，已经跑完的部分就是最重要的部分。

断点续跑
--------
「某格某折已完成」的判据是 summary CSV（默认 abl_rav30.csv）里已有
(tag=配置ID, fold=折号) 这一行 —— 这一行是 main.py 在干净测试算完之后才写的，
所以只要它存在，就说明这一折训练+验证+测试都走完了。
直接重新执行本脚本即可续跑，不会重复训练。

用法
----
    python run_abl_all.py                 # 跑全部
    python run_abl_all.py --panel 1       # 只跑 Panel 1
    python run_abl_all.py --id P1-2       # 只跑某一格
    python run_abl_all.py --dry_run       # 只打印待办清单
"""

import argparse
import csv
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_abl_ravdess import ABLATIONS, CONFIG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, 'logs', 'abl')

# 优先级：Panel 顺序即重要性顺序；Panel 内按 Panel 1 的因子设计顺序
PANEL_ORDER = [1, 2, 3, 4]


def done_set(csv_path):
    """返回已完成的 (tag, fold) 集合"""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            name = row.get('store_name', '')
            tag = name.split('_')[-1] if name else ''
            done.add((tag, str(row.get('fold', ''))))
    return done


def todo(panel=None, only_id=None):
    csv_path = os.path.join(HERE, CONFIG['summary_csv'])
    done = done_set(csv_path)
    items = []
    for p in PANEL_ORDER:
        if panel and p != panel:
            continue
        for c in ABLATIONS:
            if c['panel'] != p:
                continue
            if only_id and c['id'] != only_id:
                continue
            if c.get('alias'):
                continue
            for fold in range(CONFIG['n_folds']):
                if (c['id'], str(fold)) in done:
                    continue
                items.append((c, fold))
    return items


def run_one(cfg, fold):
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, '%s_fold%d.log' % (cfg['id'], fold))
    cmd = [sys.executable, os.path.join(HERE, 'run_abl_ravdess.py'),
           '--id', cfg['id'], '--folds', str(fold)]
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return rc, dt, log


def fmt(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return '%dh%02dm' % (h, m) if h else '%dm%02ds' % (m, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', type=int, default=0)
    ap.add_argument('--id', default='')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    items = todo(panel=args.panel or None, only_id=args.id or None)
    total = len(items)
    print('待跑: %d 个 (配置 x 折)' % total)
    for c, fold in items:
        print('  %-6s fold%d  %s' % (c['id'], fold, c['note']))
    if args.dry_run or total == 0:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    fail_log = os.path.join(LOG_DIR, 'failures.txt')
    t_start = time.time()
    for i, (c, fold) in enumerate(items, 1):
        print('\n[%d/%d] %s fold%d  %s' % (i, total, c['id'], fold, c['note']), flush=True)
        rc, dt, log = run_one(c, fold)
        status = 'OK' if rc == 0 else 'FAIL(rc=%d)' % rc
        print('     %s  用时 %s  日志 %s' % (status, fmt(dt), log), flush=True)
        if rc != 0:
            with open(fail_log, 'a') as f:
                f.write('%s fold%d rc=%d log=%s\n' % (c['id'], fold, rc, log))

        elapsed = time.time() - t_start
        eta = elapsed / i * (total - i)
        print('     已用 %s，预计剩余 %s' % (fmt(elapsed), fmt(eta)), flush=True)

    print('\n全部结束。总用时 %s' % fmt(time.time() - t_start))
    if os.path.exists(fail_log):
        print('有失败项，见', fail_log)


if __name__ == '__main__':
    main()
