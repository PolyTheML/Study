# Mistakes log

Every wrong attempt, what it revealed, and the fix. This is the most honest record of learning in the repo.

| Date | Block | What I wrote | Symptom | The misconception | The fix |
|---|---|---|---|---|---|
| 2026-10-08 | 01 | `encode("café", stoi)` | `KeyError: 'é'` | Assumed `encode` could take any text. The vocabulary only holds the 65 characters seen in the corpus, so any other character has no id. | Not a bug in the code: a closed character vocabulary cannot encode what it never saw. Block 14 (byte-level BPE) removes the problem. |
