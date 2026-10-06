"""Train the tiny GPT from the command line (graduated from notebook 11).

Suggested interface (yours to change):
    python scripts/train_gpt.py --data data/tinyshakespeare.txt --d_model 128 --n_layers 4 --steps 3000 --seed 0

It should:
  * load and encode the data, split train / val
  * build TinyGPT, train with Adam (betas 0.9, 0.98, eps 1e-9; §5.3) and noam_lr through LambdaLR (Eq. 3)
  * log train loss and validation NLL without label smoothing
  * save a checkpoint to checkpoints/ and a JSON with the config and final metrics to experiments/
"""

raise SystemExit("Block 11: your turn")
