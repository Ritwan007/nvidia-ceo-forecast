# NVIDIA CEO forecast — make.fun entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95), scored by mean pinball loss.

`main.py` (Python 3, standard library only) reads each case file in the input
directory and writes one forecast file per case to the output directory.

Model v3: the median stays at the checkpoint price (every drift and
post-spike fade term fitted on the practice window hurt out of sample).
The band scale is a recent-regime sigma (RMS of 1-minute changes over the
last 120 minutes, shrunk toward a global base, times a volatility-
clustering ratio), with a full-history-RMS rule for young markets. The
sqrt(h) widening is capped by a fitted variance term structure (empirical
15/60/240-minute change dispersion). The one change from v2: the inner
quantile levels use a shape factor that depends on the market's price
level — tightly compressed for near-dead longshots under 0.02, moderate
for mid prices, and wider than v2's uniform compression for the actively
traded favorite at 0.15+, whose interquartile range is genuinely broad.
Practice pinball loss 0.263 vs v2's 0.270 and the house baseline's 0.364;
the change also wins on in-history pseudo-checkpoint backtests and on
bootstrapped jump-heavy paths.
