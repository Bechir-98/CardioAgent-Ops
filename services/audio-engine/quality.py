"""Quality audit: corrupt / short / flat / clipped counts per class.

Stdlib wave + numpy only. Header read first (fast), signal read only if needed.
Cleaning rule for report: drop corrupt + flat, keep short (padded to 2s by
preprocess.load_wav), keep clipped (note as noisy, model must be robust).
"""
import contextlib
import wave
from pathlib import Path

import numpy as np

MIN_DUR = 2.0  # short = needs padding, not a reject
FLAT_STD = 1e-4
CLIP_LEVEL, CLIP_FRAC = 0.99, 0.01


def audit_file(path):
    """Return (status, dur): status in ok/corrupt/short/flat/clipped (first hit wins)."""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            n, fr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(n)
        dur = n / float(fr)
    except Exception:
        return "corrupt", 0.0
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dtype is None:
        return "corrupt", dur
    y = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 1:
        y = (y - 128.0) / 128.0
    elif sw == 2:
        y /= 32768.0
    else:
        y /= 2147483648.0
    if ch > 1:
        y = y.reshape(-1, ch).mean(axis=1)
    if y.std() < FLAT_STD:
        return "flat", dur
    if (np.abs(y) > CLIP_LEVEL).mean() > CLIP_FRAC:
        return "clipped", dur
    if dur < MIN_DUR:
        return "short", dur
    return "ok", dur


def audit(data="dataset/train"):
    data = Path(data)
    stats, durs = {}, []
    for cls in ("normal", "abnormal"):
        stats[cls] = {"ok": 0, "corrupt": 0, "short": 0, "flat": 0, "clipped": 0}
        sub = data / cls
        files = sorted(sub.glob("*.wav")) if sub.is_dir() else sorted(data.glob(f"{cls}*.wav"))
        for p in files:
            s, d = audit_file(p)
            stats[cls][s] += 1
            durs.append(d)
    tot = sum(sum(v.values()) for v in stats.values())
    rej = sum(v["corrupt"] + v["flat"] for v in stats.values())
    print(f"{'class':<9}{'total':>7}{'ok':>7}{'short':>7}{'clip':>7}{'flat':>7}{'corrupt':>9}")
    for cls in ("normal", "abnormal"):
        v = stats[cls]
        print(f"{cls:<9}{sum(v.values()):>7}{v['ok']:>7}{v['short']:>7}"
              f"{v['clipped']:>7}{v['flat']:>7}{v['corrupt']:>9}")
    if durs:
        print(f"total={tot} reject(corrupt+flat)={rej} ({100.0 * rej / max(tot, 1):.2f}%) "
              f"dur min/med/max={min(durs):.2f}/{float(np.median(durs)):.2f}/{max(durs):.2f}s")
    else:
        print(f"total={tot} reject={rej} (no wavs found under {data})")
    return stats


if __name__ == "__main__":  # ponytail: runnable self-check
    import sys
    s = audit(sys.argv[1] if len(sys.argv) > 1 else "data/samples")
    n = sum(sum(v.values()) for v in s.values())
    assert n > 0, "no wavs audited"
    print("ok, audited", n)
