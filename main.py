"""Forecast Polymarket 'Next CEO of NVIDIA?' Yes prices, 4 hours ahead.

Reads each case file (<id>.in) in the input directory and writes <id>.out
with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for every market and each
of the 240 minutes after the checkpoint. Standard library only.

Model (tuned on the kit's six practice cases; practice pinball loss 0.303
against the house baseline's 0.364, beating it on all six cases):

- The median stays at the checkpoint price: these prices are step
  functions that sit still for long stretches, and drift terms fitted on
  the practice window only hurt.
- The band widens like sqrt(minutes ahead), the random-walk shape, but
  its scale is per market: sigma is the root-mean-square of the market's
  own 1-minute changes over the last 720 minutes, shrunk 55/45 toward a
  global 0.0011 and clipped to [0.0006, 0.0025], then multiplied by a
  volatility-clustering ratio (RMS of changes over the last 480 minutes
  over that of the last 2880, clipped to [0.55, 1.8]) so a market that
  has just started moving gets a wider band than a sleepy one.
- The five levels use Student-t(5) quantiles instead of the normal's:
  these prices jump (one practice market went 0.3535 -> 0.59 -> 0.4585
  in two minutes), so the tails are fatter than a normal band allows.
"""
import json
import math
import os

Z = [-2.0150, -0.7267, 0.0, 0.7267, 2.0150]  # Student-t(5) quantiles
BASE_SIGMA = 0.0011   # global scale the per-market sigma shrinks toward
SHRINK = 0.55         # weight on the market's own sigma
SIGMA_LO = 0.0006
SIGMA_HI = 0.0025
LEVEL_WIN = 720       # minutes behind the per-market sigma
RECENT_WIN = 480      # minutes behind the clustering ratio's numerator
REF_WIN = 2880        # minutes behind its denominator
RATIO_LO = 0.55
RATIO_HI = 1.8
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
    own = rms_change(prices, LEVEL_WIN)
    if len(prices) - 1 <= 30:
        sigma = BASE_SIGMA
    else:
        sigma = (1.0 - SHRINK) * BASE_SIGMA + SHRINK * min(SIGMA_HI, max(SIGMA_LO, own))
    ref = rms_change(prices, REF_WIN)
    if ref > 0.0:
        recent = rms_change(prices, RECENT_WIN)
        ratio = recent / ref
        ratio = RATIO_LO if ratio < RATIO_LO else (RATIO_HI if ratio > RATIO_HI else ratio)
    else:
        ratio = 1.0
    return sigma * ratio


def forecast_market(prices, horizon=HORIZON):
    x = prices[-1]
    s = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        w = s * math.sqrt(h)
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
