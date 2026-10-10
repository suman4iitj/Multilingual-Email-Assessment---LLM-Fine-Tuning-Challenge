# Multilingual email assessment: grammar, content and CEFR

This repository predicts three scores for an email written in Polish, Italian, Greek, Dutch or Portuguese: a grammar score (0–5), a content score (0–4) and a CEFR level (A1–C2).
It was built for the SHL Hiring Assessment 2026. The method, results and error analysis are written up in the report ([PDF](docs/methodology_report.pdf), [Word](docs/methodology_report.docx)).

## Results on the held-out test set (309 emails)

| Criterion | Target | Result | 95% bootstrap interval |
|---|---|---|---|
| Grammar, Pearson r | ≥ 0.60 | 0.655 | 0.59 – 0.71 |
| Content, Pearson r | ≥ 0.60 | 0.651 | 0.58 – 0.71 |
| CEFR, Pearson r (on the predicted class) | ≥ 0.60 | 0.638 | 0.57 – 0.70 |
| CEFR, exact match | ≥ 30% | 34.6% | 29.4% – 40.1% |

All four targets are met on this file, but the margins are small and the lower ends of the intervals are slightly under the targets.
Please read the section "How the system was chosen" below before relying on these numbers.
`python src/evaluate.py --joined results/test_predictions.csv --bootstrap 2000 --by_language --confusion` recomputes them from the saved predictions.
The numbers were reproduced end to end on 10 October 2026 by loading the saved weights on a fresh Colab A100 and running both judges again; the printed output is in [`results/colab_run_log.txt`](results/colab_run_log.txt).

## How it works

1. **Fine-tuned encoder.** XLM-RoBERTa base reads (task statement, email) and predicts grammar, content and an ordinal CEFR level.
2. **Two zero-shot LLM judges.** Qwen2.5-14B-Instruct and Qwen3-14B each read the email and the grading rubric and give a score. These models are not fine-tuned.
3. **Fixed-weight blend.** The encoder and judge scores are combined (30% encoder, 70% judges), a small length prior is added to content, and CEFR also uses the combined grammar and content scores.

The judges carry most of the accuracy. On the test file the encoder alone reaches 0.44 / 0.43 / 0.41 (grammar / content / CEFR); the two judges alone reach 0.65 / 0.62 / 0.63. The full comparison is in `results/test_results_all_variants.json` and Table 2 of the report.

## Repository layout

```
src/config.py             all settings used for the reported run
src/data_synthetic.py     scenarios, label design, LLM generation, augmentation
src/data_real.py          downloads the public real texts (UniversalCEFR) and rebuilds the real split
src/model.py              multitask encoder (grammar, content, CORAL CEFR head)
src/train_encoder.py      training loop
src/judge.py              zero-shot rubric judges (digit-probability scores, cached)
src/combine.py            the final blend (pure function, unit-tested)
src/predict.py            full pipeline: encoder + judges + blend (uses the modules above)
src/predict_standalone.py the same pipeline in one file; this is the script run for the reported numbers
src/inference_encoder.py  encoder-only inference (fast, less accurate)
src/evaluate.py           metrics, bootstrap intervals, per-language table, confusion matrix
data/train_synthetic.csv  the 2,927 synthetic training emails (see data/README.md)
results/                  test metrics for every variant, predictions, validation history, judge scores, Colab run log
notebooks/SHL_Email_Assessment_v4_Final.ipynb   data generation, training and evaluation, with outputs
notebooks/Run_on_Colab.ipynb                    loads the trained weights and scores the test file
tests/                    unit and smoke tests (no GPU or downloads needed)
docs/                     methodology report (PDF and Word)
```

## Quick start

The reported run used Google Colab with one A100 40 GB GPU. `requirements.txt` lists the package versions of that environment.

### A. Score a file with the trained weights (Colab, recommended)

1. Download the trained model folder (`weights.pt`, `config.json`, tokenizer files): `<ADD LINK TO WEIGHTS>`. Put it on your Google Drive together with the test file.
2. Open [`notebooks/Run_on_Colab.ipynb`](notebooks/Run_on_Colab.ipynb) in Colab, choose an A100 runtime, edit the three paths in step 3 and run all cells.

The first run downloads and runs both 14B judges (about 25 minutes). Judge scores are cached, so a second run is fast. The notebook ends with the four metrics, a per-language table and the CEFR confusion matrix.

The same thing from a terminal on any machine with a 40 GB GPU:

```bash
git clone https://github.com/<YOUR_USERNAME>/shl-email-assessment.git && cd shl-email-assessment
pip install -r requirements.txt
python src/predict_standalone.py --input test_dataset.xlsx --model_dir path/to/model --output preds.csv --cache judge_cache.json
```

On Colab install only `transformers==5.18.0 accelerate==1.15.0 datasets==4.8.5 sentencepiece==0.2.2 openpyxl==3.1.5` and keep Colab's own `torch`, `numpy` and `pandas`; do not use `pip install -U`.

The input file needs a `text` column; `questionStatement` and `language` are used by the judges and are strongly recommended. If it also has the gold columns `grammar`, `content` and `cefr`, the script prints the metrics.

### B. Other entry points

```bash
python -m pytest tests -q                       # checks the blend and the pipeline plumbing on CPU, no downloads

# the same pipeline as modules
python src/predict.py --input test_dataset.xlsx --model_dir path/to/model --output preds.csv
python src/evaluate.py --pred preds.csv --gold test_dataset.xlsx --bootstrap 2000 --by_language --confusion

# retrain the encoder from the training CSV (needs a GPU; downloads the real texts)
python src/train_encoder.py --train data/train_synthetic.csv      # writes outputs/model
```

To regenerate the synthetic data or rerun the whole study, open `notebooks/SHL_Email_Assessment_v4_Final.ipynb` in Colab. It contains the scenarios, the generation code, training, the judges and the error analysis, with the outputs of the reported run.

Notes:

* Each 14B judge needs about 30 GB of GPU memory in bf16. A 40 GB A100 is enough because the judges are loaded one after the other.
* Judge scores are cached per input file (keyed by a hash of the emails), so a second run on the same file skips the LLMs.
* `src/predict.py --encoder_only` skips the judges. It is faster and clearly less accurate (0.44 / 0.43 / 0.41).
* The blend standardises scores over the batch it is scoring, as the reported run did on the 309 test emails. For a few emails at a time, run once on a reference batch with `--save_stats stats.json` and then pass `--stats stats.json`. With fewer than 50 emails and no saved statistics the script prints a warning.
* GPU runs are not bit-exact, so the last digit of a score can differ between machines.

### Model weights

`weights.pt` is about 1.1 GB and is not stored in git. Download it from `<ADD LINK TO WEIGHTS>` (the folder must contain `weights.pt`, `config.json` and the tokenizer files written by `save_pretrained`). It is the exponential-moving-average checkpoint of the best epoch of the reported run. If you retrain with `train_encoder.py` you get an equivalent but not identical file.

## Data

* **Synthetic (included).** 24 task scenarios written by me, five required points each. Target CEFR, grammar and content scores are sampled first; Qwen2.5-7B-Instruct then writes an email in that language that fits the targets, with the rubric's grammar descriptions as style instructions. 2,500 emails (500 per language), plus 294 rule-corrupted and 133 truncated variants, 2,927 rows. Details and column descriptions are in `data/README.md`.
* **Real (not included).** 1,050 texts from UniversalCEFR corpora: MERLIN (Italian, CC BY-SA 4.0), COPLE2 and PEAPL2 (Portuguese) and ELG-CEFR (Dutch), the last three CC BY-NC-SA 4.0. 892 are used for training, 158 are held out. They only have CEFR labels. `src/data_real.py` downloads them. If you reuse them, follow the licences and cite the corpora.
* No Polish or Greek real data was available in that collection.
* The test file was not used to write scenarios, prompts or training data.

## Model and training configuration

| | |
|---|---|
| Encoder | `xlm-roberta-base`, input = (task statement, email), max 512 tokens, masked mean pooling |
| Heads | grammar and content: two-layer MLP, targets scaled to 0–1. CEFR: CORAL ordinal head, one shared score and 5 ordered thresholds; it also receives the detached grammar and content predictions |
| Loss | 4 × MSE on grammar and content (skipped for rows without those labels) + binary cross-entropy over the 5 cumulative CEFR thresholds |
| Optimiser | AdamW, lr 2e-5, layer-wise decay 0.9, heads at 20 × lr, weight decay 0.01 |
| Schedule | 10% warm-up, then linear decay; batch size 16; 6 epochs; gradient clip 1.0 |
| Other | bf16, exponential moving average of weights (0.98), seed 42 |
| Validation | 4 of 24 scenarios held out as a group, plus the 158 real texts (CEFR only); best epoch by the sum of correlations |
| Decoding | CEFR class = number of thresholds with p > 0.5; expected level = sum of the probabilities |
| Judges | Qwen2.5-14B-Instruct, Qwen3-14B (non-thinking); 2 prompt wordings × 3 rubrics each; score = expected value over the answer-digit probabilities |
| Blend | weight 0.7 on the judges, 0.3 on the encoder; length prior 0.2 on content; CEFR = 0.5 level signal + 0.25 grammar + 0.25 content |

## Evaluation and leakage

The test file was never used for training, was never part of a prompt, and did not influence the scenarios. The judge prompts contain only the task statement, the email and shortened rubric text from the assignment.

### How the system was chosen

I built the system over several versions and scored each version on the test file, about nine scorings in total, so the test numbers are not those of an untouched set. These four choices were made after seeing test results: the 0.7/0.3 blend weights, the length prior of 0.2, replacing a 7B judge with Qwen3-14B, and the CEFR mix of 0.5/0.25/0.25 (I had also looked at how gold grammar, content and CEFR correlate on the test file). Earlier steps such as adding the judges, the MSE loss and the real data were taken with test scores in view too.
Table 2 of the report shows the alternatives side by side. A plain 50/50 blend, which I would have used without tuning, scores 0.623 / 0.603 / 0.611 and 38.2% exact match, so it also meets all four criteria, with smaller margins.

## Main limitations

* Only the encoder is fine-tuned. The judges are used zero-shot; fine-tuning one of them is the next step I would try.
* On the test file the system rarely predicts the ends of the CEFR scale (no A1 or C2, nine C1). Correlation is fine because the ordering is right, but A1, A2, C1 and C2 are mostly predicted as B1 or B2.
* The encoder scores 0.86–0.88 on its synthetic validation split and about 0.43 on real test emails. Synthetic emails differ from real learner emails, and I had no human raters to check the synthetic labels.
* Dutch and Portuguese grammar (0.53, 0.51) and Greek exact match (23%) are the weakest per-language results. With 50–80 emails per language these numbers are noisy.
* Running both judges on every email is expensive: 12 passes of a 14B model per email.

## Licences

Code: MIT (see `LICENSE`). The synthetic data was generated with Qwen2.5-7B-Instruct, and the judges are Qwen2.5-14B-Instruct and Qwen3-14B; check their model licences before reuse. The real texts are not redistributed; their licences are listed above and in `data/README.md`.
