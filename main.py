"""Forecast Polymarket 'Next CEO of NVIDIA?' Yes prices, 4 hours ahead.

Reads each case file (<id>.in) in the input directory and writes <id>.out
with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for every market and each
of the 240 minutes after the checkpoint. Standard library only.

Model v2 (rebuilt on the kit's six practice cases; practice pinball loss
0.270 against the house baseline's 0.364, beating the baseline and v1's
0.303 on all six cases; leave-one-case-out mean 0.270):

- The median stays at the checkpoint price: these prices are step
  functions that sit still for long stretches, and every drift, momentum,
  mean-reversion and post-spike fade term fitted on the practice window
  hurt out of sample, so the center does not move.
- The band scale is a *recent-regime* sigma: the RMS of the market's own
  1-minute changes over the last 120 minutes, shrunk 35/65 toward a global
  0.0011, clipped to [0.0002, 0.006], times the volatility-clustering
  ratio RMS(480m)/RMS(2880m) clipped to [0.55, 1.8]. A long estimation
  window keeps one old news spike inflating the band for hours; the
  recent window tracks the regime the market is actually in.
- Young markets (at most 30 observed changes) cannot estimate a 120-minute
  RMS, so they use their full-history RMS floored at 0.0011 and capped at
  0.012 instead of the quiet global base: the one such practice market
  ranged over 0.19 inside its forecast window.
- The sqrt(h) widening is capped by a fitted variance term structure
  (research T6, after Lo & MacKinlay's variance ratio): the empirical
  standard deviation of 15/60/240-minute changes over the last 1080
  minutes implies a per-root-minute scale s(h); where s(h) says variance
  grows slower than h (bounded prices mean-revert), the band follows it.
  The cap only ever narrows the band, never widens it.
- The five levels use Student-t(5) quantiles (these prices jump, so the
  tails are fatter than a normal band allows), with the two inner levels
  compressed by 0.7: at short horizons most minutes do not move at all,
  so the interquartile range of the step process is far tighter than a
  scaled t shape implies.
"""
import json
import math
import os

Z = [-2.0150, -0.50869, 0.0, 0.50869, 2.0150]  # t(5); inner levels x0.7
BASE_SIGMA = 0.0011   # global scale the per-market sigma shrinks toward
SHRINK = 0.65         # weight on the market's own recent sigma
SIGMA_LO = 0.0002
SIGMA_HI = 0.006
LEVEL_WIN = 120       # minutes behind the regime sigma
RECENT_WIN = 480      # minutes behind the clustering ratio's numerator
REF_WIN = 2880        # minutes behind its denominator
RATIO_LO = 0.55
RATIO_HI = 1.8
YOUNG_MAX_CHANGES = 30
YOUNG_HI = 0.012
TS_WIN = 1080         # window behind the term-structure anchors
TS_ANCHORS = (15, 60, 240)
HORIZON = 240


def rms_change(prices, win):
    """Root-mean-square of 1-minute changes over the last `win` minutes."""
    seg = prices[-win:] if len(prices) > win else prices
    if len(seg) < 2:
        return 0.0
    acc = 0.0
    prev = seg[0]
    for v in seg[1:]:
        d = v - prev
        acc += d * d
        prev = v
    return math.sqrt(acc / (len(seg) - 1))


def market_sigma(prices):
    """Recent-regime band scale for one market (see module docstring)."""
    if len(prices) - 1 <= YOUNG_MAX_CHANGES:
        own = rms_change(prices, len(prices))
        return BASE_SIGMA if own < BASE_SIGMA else (YOUNG_HI if own > YOUNG_HI else own)
    own = rms_change(prices, LEVEL_WIN)
    own = SIGMA_LO if own < SIGMA_LO else (SIGMA_HI if own > SIGMA_HI else own)
    sigma = (1.0 - SHRINK) * BASE_SIGMA + SHRINK * own
    ref = rms_change(prices, REF_WIN)
    if ref > 0.0:
        ratio = rms_change(prices, RECENT_WIN) / ref
        ratio = RATIO_LO if ratio < RATIO_LO else (RATIO_HI if ratio > RATIO_HI else ratio)
        sigma *= ratio
    return sigma


def step_std(prices, step, win):
    """Population std of `step`-minute changes over the last `win` prices."""
    seg = prices[-win:] if len(prices) > win else prices
    if len(seg) < step + 10:
        return None
    d = [seg[i + step] - seg[i] for i in range(len(seg) - step)]
    mean = sum(d) / len(d)
    var = sum((v - mean) * (v - mean) for v in d) / len(d)
    return math.sqrt(var)


def term_structure_caps(prices):
    """Per-horizon width caps from the fitted variance term structure.

    Returns a list of (h, cap) breakpoints on a log-log scale of the
    implied per-root-minute scale, or None when too few anchors exist.
    """
    pts = []
    for a in TS_ANCHORS:
        sd = step_std(prices, a, TS_WIN)
        if sd is not None and sd > 0.0:
            pts.append((math.log(a), math.log(sd / math.sqrt(a))))
    if len(pts) < 2:
        return None
    return pts


def ts_scale(pts, h):
    """Log-log interpolated per-root-minute scale at horizon h (clamped)."""
    lh = math.log(h)
    if lh <= pts[0][0]:
        return math.exp(pts[0][1])
    if lh >= pts[-1][0]:
        return math.exp(pts[-1][1])
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= lh <= x1:
            t = (lh - x0) / (x1 - x0)
            return math.exp(y0 + t * (y1 - y0))
    return math.exp(pts[-1][1])


def forecast_market(prices, horizon=HORIZON):
    x = prices[-1]
    s = market_sigma(prices)
    pts = term_structure_caps(prices)
    rows = []
    for h in range(1, horizon + 1):
        w = s * math.sqrt(h)
        if pts is not None:
            cap = ts_scale(pts, h) * math.sqrt(h)
            if cap < w:
                w = cap
        row = []
        prev = 0.0
        for z in Z:
            v = x + z * w
            v = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
            if v < prev:
                v = prev
            prev = v
            row.append(round(v, 6))
        rows.append(row)
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecasts = []
        for m in case["markets"]:
            forecasts.append({"id": m["id"], "q": forecast_market(m["prices"], int(case.get("horizon_minutes", HORIZON)))})
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
