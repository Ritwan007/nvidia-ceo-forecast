# NVIDIA CEO forecast — make.fun entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95), scored by mean pinball loss.

`main.py` (Python 3, standard library only) reads each case file in the input
directory and writes one forecast file per case to the output directory.

Model v4: the median stays at the checkpoint price (every drift and
post-spike fade term fitted on the practice window hurt out of sample).
The band scale is a recent-regime sigma (RMS of 1-minute changes over the
last 120 minutes, shrunk toward a global base, times a volatility-
clustering ratio), with a full-history-RMS rule for young markets. The
sqrt(h) widening is capped by a fitted variance term structure (empirical
15/60/240-minute change dispersion), and the inner quantile levels use a
shape factor that depends on the market's price level — tightly compressed
for near-dead longshots under 0.02, moderate for mid prices, and wider for
the actively traded favorite at 0.15+, whose interquartile range is
genuinely broad.

Four additions over v3, each small alone and stackable: a market that
cannot fit its own term structure borrows the cap from its price-level
bucket's median variance ratio; the outer quantile facing the nearer
price boundary (0 or 1) is compressed, since that tail's room does not
exist; a fresh jump in the last 15 minutes temporarily widens the band
with a 30-minute decay (the center never fades); and the whole band
shifts down by an eighth of its width, because in a 24-way market most
prices bleed toward zero between checkpoints.

Practice pinball loss 0.254 vs v3's 0.263, v2's 0.270 and the house
baseline's 0.364. The improvement also holds in leave-one-case-out, on
7,892 in-history walk-forward windows (paired mean −0.011 vs v3, 95%
block-bootstrap CI entirely below zero), and on 25,000 bootstrapped
jump-heavy paths.
