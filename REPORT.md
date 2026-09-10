# Supplementary report

This file used to carry a second copy of the map from paper elements to result files, and it drifted
away from `REPRODUCE.md`. The map now lives in one place only.

See **[REPRODUCE.md](REPRODUCE.md)**, which lists, for every element of the main text and of the
supplement (sections S1-S6, Tables S1-S16, Fig. S1): the result file it comes from, the command that
regenerates that file, the measured runtime of that command, and the macro family or generated table
body it appears in. It also records the fixed design choices - calibration split, metric, dead-label
policies, bootstrap, threshold grid, cap sweep, MULAN learners and weights, and the float32 storage
of the prediction arrays - and what in this repository cannot be recomputed.
