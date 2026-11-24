# Submissions (files here are what we wrote; "submitted" = sent to Kaggle)

| File | What | val CV | Kaggle public |
|---|---|---|---|
| `sub_baseline.csv` | TF-IDF baseline | 0.866 | 0.86928 |
| `sub_stack.csv` | early logistic-regression stack (a few models) | — | (see notes.md) |
| `sub_stack_lgb_ctx_v1.csv` | 4 models, LightGBM meta + lang/domain | 0.9546 | 0.95148 |
| `sub_stack_lgb_ctx_v2_6models_submitted.csv` | 6 models | 0.9591 | 0.95693 |
| `sub_stack_lgb_ctx_8models.csv` | 8 models (not submitted) | 0.9644 | — |
| `sub_stack_lgb_ctx_9models_winmax_submitted.csv` | 9 models + window-max | 0.9703 | 0.96844 |
| `sub_stack_lgb_ctx_10models_winmax.csv` | 10 models + window-max of two windowed encoders | 0.9736 | 0.97167 |
| `sub_stack_lgb_ctx_12models_blogdeb_enc8.csv` | 12 models (+ blogdeb, enc8) | 0.9767 | 0.97551 |
| `sub_stack_lgb_ctx_12models_lean.csv` | lean 12 models (no enc / encwin / encwin8) | 0.9770 | 0.97643 |
| `sub_stack_lgb_ctx_15models.csv` | 15 models (+ qwen9, encwin8, encwin10) | 0.9778 | **0.97788** |

**Final 2 (to select on Kaggle): `sub_stack_lgb_ctx_15models.csv` (58743317) and `sub_stack_lgb_ctx_12models_lean.csv` (58788689).**

New files are written by `src/stack.py --write` to `submissions/sub_stack_<meta>.csv` (e.g. `sub_stack_lgb_ctx.csv`); rename them before overwriting.
Format: 25,000 rows `id,label`, order of `data/raw/sample_submission.csv`, labels in {human, ai, fusion}.
