"""Forecast Polymarket 'Next CEO of NVIDIA?' Yes prices, 4 hours ahead.

Reads each case file (<id>.in) in the input directory and writes <id>.out
with five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for every market and each
of the 240 minutes after the checkpoint. Standard library only.

Model v4 (v3 plus four changes, each validated on the six practice cases,
in leave-one-case-out, on 7,892 in-history walk-forward windows with a
block-bootstrap confidence interval, and on 25,000 bootstrapped
jump-heavy paths; practice pinball loss 0.254 against the house
baseline's 0.364, v2's 0.270 and v3's 0.263):

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
  (after Lo & MacKinlay's variance ratio): the empirical standard
  deviation of 15/60/240-minute changes over the last 1080 minutes implies
  a per-root-minute scale s(h); where s(h) says variance grows slower
  than h (bounded prices mean-revert), the band follows it. The cap only
  ever narrows the band, never widens it.
- The outer levels use Student-t(5) quantiles (these prices jump, so the
  tails are fatter than a normal band allows). The inner levels use a
  per-price-level shape factor times the normal 0.6745 spacing: at short
  horizons most minutes do not move at all, so the interquartile range of
  the step process is far tighter than any fixed scaled shape implies --
  and how much tighter depends on the price level. A near-dead longshot
  under 0.02 moves in whole ticks or not at all (factor 0.20); mid-priced
  markets sit between (0.68); the actively traded favorite at 0.15+ moves
  almost every minute, so its interquartile range is *wider* than the
  compressed shape v2 used everywhere (1.60). The factors are pooled
  across markets per price-level bucket (the one cross-market effect the
  literature supports: pooled shrinkage, not per-market fits), and the
  surface around them is a flat plateau, not a knife edge.
- NEW (pooled term-structure fallback): a market too young or too quiet
  to fit its own term structure (fewer than two usable anchors) borrows
  the cap from its price-level bucket: the bucket's median variance
  ratio at each anchor, rescaled by the market's own 1-minute RMS. The
  borrowed cap only ever narrows, exactly like the own cap. On the
  practice set this is worth most on the youngest market, whose own
  history cannot express how slowly variance accumulates.
- NEW (boundary-compressed outer tail): the outer quantile on the side
  facing the nearer boundary is compressed by up to 40% as the price
  approaches 0 or 1 (multiplier ramping from 0.6 at the boundary to 1.0
  at a distance of 0.05). A price at 0.01 cannot fall 0.05; the symmetric
  t(5) tail pays pinball rent for room that does not exist. The open
  side is untouched -- widening it was tested and hurts.
- NEW (post-jump width gate): if the last 15 minutes' RMS exceeds the
  last 120 minutes' RMS, the excess (capped at +100%) temporarily widens
  the band, decaying with a 30-minute time constant,
  w *= sqrt(1 + g * exp(-h/30)). Fresh jumps cluster; the widening is
  confined to the horizons where the clustering lives, and the center
  does not move (fading the center was falsified in v2).
- NEW (downward band shift): all five quantiles shift down by 0.125
  band-widths. In a 24-way winner-take-all market most prices bleed
  toward zero between checkpoints -- only one market can win -- so the
  conditional distribution of the typical market's next four hours sits
  slightly below its checkpoint price, and a symmetric band centred on
  the checkpoint systematically overprices its upper half. The shift is
  in width units, so it scales with each market's own volatility, and
  the response surface around it is a plateau (0.10-0.20 all score
  within noise of each other out of sample).
"""
import json
import math
import os
import statistics

Z_OUTER = 2.0150      # Student-t(5) 0.95 quantile
Z_INNER = 0.674490    # normal 0.75 quantile; scaled by the bucket factor
# Inner shape factor by checkpoint price level: (upper edge, factor).
INNER_BUCKETS = ((0.02, 0.20), (0.15, 0.68), (float("inf"), 1.60))
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
# v4 additions (see docstring; each one is small alone, they stack)
SHIFT = -0.125        # band shift, in units of the post-cap width
ASYM_A = 0.6          # boundary-side outer multiplier at the boundary
ASYM_RAMP = 0.05      # distance over which it ramps back to 1.0
GATE_TAU = 30.0       # post-jump widening decay, minutes
GATE_MAX = 1.0        # cap on the gate's excess-RMS ratio


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


def inner_factor(x):
    """Inner-quantile shape factor for a market priced at x."""
    for edge, f in INNER_BUCKETS:
        if x < edge:
            return f
    return INNER_BUCKETS[-1][1]


def bucket3(x):
    """Price-level bucket: 0 = longshot, 1 = mid, 2 = active favorite."""
    return 0 if x < 0.02 else (1 if x < 0.15 else 2)


def vr_anchors(prices):
    """Variance-ratio anchors VR(a) = sd(a-min changes)/(sqrt(a) * RMS1)."""
    seg = prices[-TS_WIN:] if len(prices) > TS_WIN else prices
    if len(seg) < 20:
        return None
    r1 = rms_change(seg, len(seg))
    if r1 <= 0.0:
        return None
    out = {}
    for a in TS_ANCHORS:
        sd = step_std(prices, a, TS_WIN)
        if sd is not None and sd > 0.0:
            out[a] = sd / (math.sqrt(a) * r1)
    return out if len(out) >= 2 else None


def build_pool(all_prices):
    """Median variance-ratio anchor per price bucket across the case."""
    by_bucket = {0: [], 1: [], 2: []}
    for prices in all_prices:
        va = vr_anchors(prices)
        if va:
            by_bucket[bucket3(prices[-1])].append(va)
    pool = {}
    for b, lst in by_bucket.items():
        if not lst:
            pool[b] = None
            continue
        d = {}
        for a in TS_ANCHORS:
            vals = [v[a] for v in lst if a in v]
            if vals:
                d[a] = statistics.median(vals)
        pool[b] = d if len(d) >= 2 else None
    return pool


def pooled_cap_widths(prices, pool, horizon):
    """Width path from the bucket-pooled term structure, or None."""
    if pool is None:
        return None
    va = pool.get(bucket3(prices[-1]))
    if not va:
        return None
    seg = prices[-TS_WIN:] if len(prices) > TS_WIN else prices
    r1 = rms_change(seg, len(seg))
    if r1 <= 0.0:
        return None
    pts = [(math.log(a), math.log(va[a] * r1)) for a in sorted(va)]
    if len(pts) < 2:
        return None
    return [ts_scale(pts, h) * math.sqrt(h) for h in range(1, horizon + 1)]


def outer_multipliers(x):
    """(lower, upper) outer-z multipliers: the boundary-facing side is
    compressed near 0/1, the open side is left at 1.0."""
    lo_room = x
    hi_room = 1.0 - x
    if lo_room <= hi_room:
        m = ASYM_A + (1.0 - ASYM_A) * min(lo_room / ASYM_RAMP, 1.0)
        return m, 1.0
    m = ASYM_A + (1.0 - ASYM_A) * min(hi_room / ASYM_RAMP, 1.0)
    return 1.0, m


def forecast_market(prices, pool=None, horizon=HORIZON):
    x = prices[-1]
    s = market_sigma(prices)
    pts = term_structure_caps(prices)
    pw = None
    if pts is None and pool is not None:
        pw = pooled_cap_widths(prices, pool, horizon)
    gate = 0.0
    base = rms_change(prices, LEVEL_WIN)
    if base > 0.0:
        gate = rms_change(prices, 15) / base - 1.0
        gate = 0.0 if gate < 0.0 else (GATE_MAX if gate > GATE_MAX else gate)
    z_in = Z_INNER * inner_factor(x)
    lo_m, hi_m = outer_multipliers(x)
    z = [-Z_OUTER * lo_m + SHIFT, -z_in + SHIFT, SHIFT, z_in + SHIFT,
         Z_OUTER * hi_m + SHIFT]
    rows = []
    for h in range(1, horizon + 1):
        w = s * math.sqrt(h)
        if pts is not None:
            cap = ts_scale(pts, h) * math.sqrt(h)
            if cap < w:
                w = cap
        elif pw is not None:
            if pw[h - 1] < w:
                w = pw[h - 1]
        if gate > 0.0:
            w *= math.sqrt(1.0 + gate * math.exp(-h / GATE_TAU))
        row = []
        prev = 0.0
        for zv in z:
            v = x + zv * w
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
        pool = build_pool([m["prices"] for m in case["markets"]])
        forecasts = []
        for m in case["markets"]:
            forecasts.append({"id": m["id"], "q": forecast_market(m["prices"], pool, int(case.get("horizon_minutes", HORIZON)))})
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
