# Study guide

## One session (30 to 90 minutes)
1. Open the block's notebook. In the Claude project chat, say which block you are on (or type `where are we`).
2. Claude teaches ONE concept: concept, intuition, math, shapes, then stops. Type `next` to continue, `deeper` to go
   back to first principles, `code` to jump to shapes and the coding step.
3. Answer the "Before you code" questions in the notebook. Guess freely.
4. Write the code yourself in the **YOUR CODE** cell. When it fails, paste the code and the output into chat. Claude
   names the misconception, explains why it was tempting, and rebuilds the idea. It does not hand you the solution.
5. Run the checkpoint cell until it prints `All checks passed`.
6. Log each mistake in the notebook's table, then in `journal/mistakes.md`.
7. Type `check` for a collaborative check-in. Fill "Explain it back" without notes.
8. Graduate the code into `src/`, run `pytest -q`, commit.
9. End: Claude tells you where you are, what is next, and one small thing to implement before the next session.
   Add a line to `journal/learning_log.md`.

## Rules that keep this honest
- Do not open `checks/` or `tests/` until your block passes. They test behavior, but reading them first turns
  learning into pattern matching.
- No copying code from tutorials or other repos while doing a block. Reading other code AFTER you pass is encouraged;
  note what you would change.
- Every experiment is pre-registered in `experiments/` before it runs: hypothesis, and the result that would falsify it.

## Running
Local:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
jupyter lab
```
Google Colab (blocks 11, 13, 16 benefit from a GPU): push the repo to GitHub first, then in the first cell
```python
!git clone https://github.com/<you>/transformer-from-scratch.git
%cd transformer-from-scratch
!pip install -q -e .
```
and open notebooks from the repo root. Commit from your own machine, not from Colab.

## Git workflow
First time (do this today, as a private repo):
```bash
cd transformer-from-scratch
git init -b main
git add .
git commit -m "scaffold: study structure, specs and checks (drafted with Claude as tutor)"
# create an empty PRIVATE repo named transformer-from-scratch on github.com, then:
git remote add origin https://github.com/<you>/transformer-from-scratch.git
git push -u origin main
```
Then:
- One commit (or a few) per block, with messages like `block 06: scaled dot-product attention`.
- Notebooks are committed WITH their outputs: the printed checks and plots are evidence of the work.
- Never commit `data/`, `checkpoints/`, or the paper PDF (see `.gitignore`).
- Push to GitHub from the start, as a private repo. Make it public once milestone M3 is done and the README has results.

## Shortcuts (from the project protocol)
`skim`, `next`, `deeper`, `code`, `sheet`, `check`, `where are we`
