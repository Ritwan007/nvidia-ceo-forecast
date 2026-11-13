# NVIDIA CEO forecast — make.fun entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after a checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95), scored by mean pinball loss.

`main.py` (Python 3, standard library only) reads each case file in the input
directory and writes one forecast file per case to the output directory.

Model: the median stays at the checkpoint price; the band around it widens
like sqrt(minutes ahead), scaled per market by the market's own recent
volatility, a short-term volatility-clustering ratio, and Student-t(5)
quantile levels (fatter tails than the normal, because these prices jump).
