# Data

Nothing in this folder is committed except this file.

| File | Source | Used in | How to get it |
|---|---|---|---|
| `tinyshakespeare.txt` | github.com/karpathy/char-rnn (public-domain Shakespeare) | blocks 01 to 13 | run notebook 00 (notebook 01 downloads it again if it is missing) |
| `train_ids.npy`, `val_ids.npy` | made by you | blocks 02 onward | notebook 01; read them back with `load_ids()` |
| Khmer-English parallel data | chosen in block 16 (check the license first) | blocks 14 to 16 | see notebook 16 |

Notebooks do not share variables, so each block hands its result to the next through these files.
Never type their paths by hand.
Import them from `minitransformer.paths`, which is anchored to the repo root and works from any working directory.

Never put confidential work documents here, even locally: notebooks get committed with their outputs.
