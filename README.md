# NVIDIA CEO forecast — make.fun entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95), scored by mean pinball loss.

`main.py` (Python 3, standard library only) reads each case file in the input
directory and writes one forecast file per case to the output directory.

Model v2: the median stays at the checkpoint price (every drift and
post-spike fade term fitted on the practice window hurt out of sample).
The band scale is a recent-regime sigma (RMS of 1-minute changes over the
last 120 minutes, shrunk toward a global base, times a volatility-
clustering ratio), with a full-history-RMS rule for young markets. The
sqrt(h) widening is capped by a fitted variance term structure (empirical
15/60/240-minute change dispersion), and the inner Student-t(5) quantile
levels are compressed to match the tight interquartile range of a step
process. Practice pinball loss 0.270 vs the house baseline's 0.364.
