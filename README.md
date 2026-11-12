# cultural_speech_project
# Cultural Speech–Semantic Joint Model

## Overview

This project provides a Python implementation for Chinese speech recognition, cultural intent classification, and cultural text generation. The acoustic network uses MFCC features, a bidirectional LSTM encoder, and a CTC output head. A blank-aware differentiable bridge maps CTC character posteriors into continuous semantic vectors, which are consumed by a semantic LSTM and a conditional generation LSTM.

The three intent classes are `cultural_inquiry`, `emotional_expression`, and `creative_request`. The generation tasks include classical poetry continuation and cultural story generation.

## Architecture and mathematical formulation

The input waveform is resampled to 16 kHz and represented by 13 static MFCC coefficients plus first- and second-order temporal differences:

$$
\mathbf{x}_t = [\operatorname{MFCC}_{13}(t);\,\Delta\operatorname{MFCC}_{13}(t);\,\Delta^2\operatorname{MFCC}_{13}(t)] \in \mathbb{R}^{39}.
$$

A three-layer bidirectional LSTM produces a context-dependent acoustic state, followed by a linear character classifier and softmax:

$$
\mathbf{h}_t = [\overrightarrow{\operatorname{LSTM}}(\mathbf{x})_t;\overleftarrow{\operatorname{LSTM}}(\mathbf{x})_t], \qquad
p_t(k)=\operatorname{softmax}(W\mathbf{h}_t+b)_k.
$$

The CTC blank posterior is separated from the distribution over output characters. With blank index $\varnothing$, non-blank confidence $r_t$, and character embedding matrix $E$, the bridge computes:

$$
r_t=1-p_t(\varnothing), \qquad
\bar p_t(k)=\frac{p_t(k)}{\max(r_t,\varepsilon)}\quad(k\neq\varnothing), \qquad
\mathbf{z}_t=r_t\sum_{k\neq\varnothing}\bar p_t(k)E_k.
$$

The semantic LSTM processes $\mathbf{z}_{1:T}$; confidence-weighted pooling produces a sentence representation for three-way intent classification and decoder initialization. The generation LSTM uses teacher forcing during training. CTC candidate re-ranking combines the acoustic log score with cosine similarity in the shared semantic space:

$$
S(y)=\log P_{\mathrm{CTC}}(y\mid x)+\lambda\operatorname{cos}(\mathbf{s}_{\mathrm{audio}},\mathbf{s}_{y}).
$$

The coefficient $\lambda$ is selected using validation samples by `scripts/tune_rerank.py`. Joint training uses:

$$
\mathcal{L}=0.60\mathcal{L}_{\mathrm{CTC}}+0.25\mathcal{L}_{\mathrm{intent}}+0.15\mathcal{L}_{\mathrm{generation}}.
$$

The CTC term is sequence negative log-likelihood, the intent term is cross-entropy over three categories, and the generation term is token-level negative log-likelihood.

## Model and training configuration

The full configuration is `configs/paper.yaml`; the following values are read from that file.

| Category | Setting | Value |
|---|---|---|
| Acoustic frontend | Sampling rate | 16,000 Hz |
| Acoustic frontend | MFCC coefficients | 13 |
| Acoustic frontend | Dynamic features | 13 delta + 13 delta-delta |
| Acoustic frontend | Combined feature size | 39 |
| Acoustic frontend | Window length | 25 ms |
| Acoustic frontend | Frame shift | 10 ms |
| Acoustic frontend | FFT size | 400 |
| Acoustic frontend | Mel filters | 40 |
| Speech encoder | Recurrent architecture | Bidirectional LSTM |
| Speech encoder | Number of layers | 3 |
| Speech encoder | Hidden size per direction | 256 |
| Speech recognition | Character output | Linear projection + CTC |
| Speech recognition | Blank token ID | 0 |
| Semantic bridge | Representation | Non-blank normalized posterior-weighted embeddings |
| Semantic encoder | Number of layers | 2 |
| Semantic encoder | Hidden size | 256 |
| Character embeddings | Embedding dimension | 128 |
| Intent classifier | Output categories | 3 |
| Generation decoder | Number of LSTM layers | 2 |
| Generation decoder | Hidden size | 256 |
| Generation decoder | Poetry maximum output length | 64 tokens |
| Generation decoder | Cultural story maximum output length | 128 tokens |
| Decoding configuration | CTC beam width | 10 |
| Decoding configuration | Semantic re-ranking candidates | 5 |
| Decoding configuration | Generation beam-width field | 5 |
| Training | Optimizer | Adam |
| Training | Learning rate | 1 × 10⁻³ |
| Training | Pretrained comparison-system fine-tuning LR | 2 × 10⁻⁵ |
| Training | Batch size | 32 |
| Training | Dropout | 0.30 |
| Training | Maximum epochs | 50 |
| Training | Early-stopping patience | 10 epochs |
| Training | Gradient clipping norm | 5.0 |
| Training | CTC loss weight | 0.60 |
| Training | Intent loss weight | 0.25 |
| Training | Generation loss weight | 0.15 |
| Independent runs | Random seeds | 17, 29, 42, 73, 101 |
| Confidence intervals | Confidence level | 95% |
| Bootstrap comparison | Number of resamples | 10,000 |

The `generation_beam_size` field is present in the configuration, while the current `evaluate.py` path calls `greedy_generate` for text output. The CTC decoder uses beam search followed by semantic candidate re-ranking.

### Computing environment

The study configuration describes Ubuntu 20.04 LTS, Python 3.10, PyTorch 2.1 with CUDA 12.1, two NVIDIA A100 GPUs with 40 GB memory each, an Intel Xeon Gold 6330 CPU, and 256 GB RAM. The edge evaluation target is a Raspberry Pi 4B with 4 GB RAM, CPU inference, and batch size 1. The project includes CPU execution and dynamic INT8 quantization profiling utilities.

## Dataset organization

| Dataset | Scope | Data split | Use |
|---|---|---|---|
| AISHELL-1 | Approximately 178 hours of Mandarin speech from 400 speakers | Official train / development / test | Mandarin ASR |
| Common Voice Chinese v16 | 44 hours after filtering | 36 h / 4 h / 4 h | Multi-dialect and noise ASR |
| THUCNews subset | 90,000 news texts | 70,000 / 10,000 / 10,000 | Auxiliary semantic text training |
| Chinese Poetry subset | 60,000 poetry sequences | 50,000 / 5,000 / 5,000 | Generation decoder text training |
| CSSP | 4,800 recorded utterances from 60 speakers | 2,880 / 960 / 960 | Speech–semantic and cultural-task evaluation |
| Field-Culture Test | 600 recorded utterances from 30 independent speakers | Held-out test only | Cross-setting evaluation |

The CSSP intent labels contain 1,600 cultural inquiries, 1,600 emotional expressions, and 1,600 creative requests, with 800 poetry-continuation and 800 story-generation requests among the creative examples. CSSP assigns speakers to training, validation, and test partitions in a 36:12:12 speaker ratio. The Field-Culture Test contains 200 utterances each from a museum exhibition hall, an intangible cultural heritage workshop, and an outdoor cultural setting; the three intent classes each contribute 200 utterances. Field-Culture speakers are disjoint from CSSP speakers.

### Data sources

- AISHELL-1: <https://www.openslr.org/33/>
- Common Voice Chinese v16 mirror: <https://huggingface.co/datasets/fsicoli/common_voice_16_0>
- THUCNews project and corpus access information: <https://github.com/thunlp/THUCTC>
- Chinese Poetry: <https://github.com/chinese-poetry/chinese-poetry>
- Field-Culture Test project: <https://github.com/C3R8U/Field-Culture-Test>
- CSSP: access through the authorized data custodian.

For datasets stored outside this package, supply actual recordings and aligned annotations before running corpus experiments; `DATA_ACCESS.md` describes source handling and access requirements. The files under `examples/synthetic_demo/` are generated electronic-tone software-test fixtures, not participant recordings. Their outputs under `runs/demo/` are software-test records, not corpus research measurements.

### Manifest format

Each speech record is a JSON object stored as one line in a `.jsonl` file. The loader expects `sample_id`, `audio_path`, `speaker_id`, `text`, `dataset`, `split`, and `origin`; cultural tasks additionally use `intent`, `scenario`, and `target_text`. `origin` is `observed` for collected recordings or `public_corpus` for downloaded corpora. Audio paths may be absolute or relative to the manifest's location.

The accepted cultural intent values are `cultural_inquiry`, `emotional_expression`, and `creative_request`. Training, development, and test speaker identities must be separated for CSSP; all Field-Culture samples belong to the held-out test partition.

## Installation

From the extracted `cultural_speech_project` directory:

```bash
python -m venv .venv
# Activate the environment using your operating system's shell conventions.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Python 3.10 is the documented training-environment version. A CUDA-enabled installation of PyTorch may be used when a compatible GPU runtime is present.

## Data preparation and training

The commands below assume locally available audio and annotation files in the paths shown. Run all commands from the project root.

### Speech and text manifests

```bash
python scripts/prepare_corpora.py aishell --root /datasets/aishell --output data/manifests
python scripts/prepare_corpora.py common_voice --root /datasets/common_voice_zh --output data/manifests
python scripts/prepare_corpora.py thucnews --root /datasets/THUCNews --output data/text
python scripts/prepare_corpora.py poetry --root /datasets/chinese-poetry --output data/text
python scripts/prepare_corpora.py cultural --dataset CSSP --csv /private/cssp.csv --audio-root /private/cssp_audio --output data/manifests
python scripts/prepare_corpora.py cultural --dataset 'Field-Culture Test' --csv /private/field.csv --audio-root /private/field_audio --output data/manifests
python scripts/validate_data.py data/manifests/cssp_train.jsonl data/manifests/cssp_dev.jsonl data/manifests/cssp_test.jsonl --output artifacts/cssp_validation.json
```

### Independent training runs and held-out evaluation

```bash
python scripts/run_seeds.py --config configs/paper.yaml --train data/manifests/cssp_train.jsonl --validation data/manifests/cssp_dev.jsonl --test data/manifests/cssp_test.jsonl --output runs/cssp
python scripts/evaluate.py --checkpoint runs/cssp/seed_17/best.pt --test data/manifests/field_culture_test.jsonl --rerank-config runs/cssp/seed_17/rerank.json --output runs/cssp/seed_17/field_culture
python scripts/aggregate_seeds.py runs/cssp/seed_*/test/evaluation.json --metric wer --output artifacts/cssp_wer_summary.json
python scripts/score_predictions.py --input runs/cssp/seed_17/test/predictions.jsonl --output artifacts/cssp_seed17_scores.json
```

The `run_seeds.py` command uses the five seeds configured by default, creates one checkpoint and training log per seed, tunes the semantic re-ranking weight on validation audio, and writes held-out prediction and evaluation files. The auxiliary text pretraining script is available for workflows with prepared THUCNews and Chinese Poetry data.

### Data checks, unit tests, and profiling

```bash
python -m pytest tests -q
python scripts/profile_edge.py --checkpoint runs/cssp/seed_17/best.pt --manifest data/manifests/cssp_test.jsonl --output artifacts/cpu_profile.json
python scripts/profile_edge.py --checkpoint runs/cssp/seed_17/best.pt --manifest data/manifests/cssp_test.jsonl --output artifacts/cpu_profile_int8.json --int8
```

The profiling utility reports latency, real-time factor, throughput, and process memory; processor-wide utilization and external power-meter readings require measurements on the target hardware.

## Metrics and reporting

ASR scoring includes character error rate (CER) and Chinese word error rate (WER), with Jieba-based word segmentation selected by `configs/paper.yaml`. Intent evaluation includes accuracy, class-wise precision, recall, F1, macro averages, support, and a three-by-three confusion matrix. Generation scoring includes character-level corpus BLEU-2, character-level ROUGE-L, and optional BERTScore. Statistical utilities cover Student-t confidence intervals across independent runs, paired bootstrap comparisons, and Friedman/Kendall-W analysis for within-subject ratings.

`training.jsonl` records training and validation loss components by epoch; `predictions.jsonl` stores sample-level model outputs; and `evaluation.json` stores metrics derived from those records. Model checkpoints retain the vocabulary, configuration, random seed, and input-source metadata.

## File reference

Each file in the package is listed below with a one-sentence description. Relative paths start at the `cultural_speech_project` directory.

| Relative path | Purpose |
|---|---|
| `CHECKSUMS.sha256` | SHA-256 digests of packaged project files for integrity checking. |
| `DATA_ACCESS.md` | Data acquisition, access conditions, and input-origin notes. |
| `README.md` | Project architecture, configuration, command usage, and file inventory. |
| `artifacts/execution_verification.json` | Machine-readable status and environment details of software-test execution. |
| `artifacts/reported_metrics_reference.json` | Published reference values recorded separately from computed experiment outputs. |
| `artifacts/software_test_metrics_by_seed.csv` | Per-seed software-test measurements in CSV form. |
| `configs/paper.yaml` | Main model, feature, optimizer, hardware, dataset, and seed configuration. |
| `configs/smoke.yaml` | Compact configuration for local software functionality checks. |
| `examples/synthetic_demo/dev.jsonl` | Generated-tone records used as the validation manifest for software tests. |
| `examples/synthetic_demo/source_metadata.json` | Source identification and settings for generated-tone fixtures. |
| `examples/synthetic_demo/test.jsonl` | Generated-tone records used as the evaluation manifest for software tests. |
| `examples/synthetic_demo/tone_000.wav` | Generated electronic-tone audio fixture 000 used to exercise audio loading. |
| `examples/synthetic_demo/tone_001.wav` | Generated electronic-tone audio fixture 001 used to exercise audio loading. |
| `examples/synthetic_demo/tone_002.wav` | Generated electronic-tone audio fixture 002 used to exercise audio loading. |
| `examples/synthetic_demo/tone_003.wav` | Generated electronic-tone audio fixture 003 used to exercise audio loading. |
| `examples/synthetic_demo/tone_004.wav` | Generated electronic-tone audio fixture 004 used to exercise audio loading. |
| `examples/synthetic_demo/tone_005.wav` | Generated electronic-tone audio fixture 005 used to exercise audio loading. |
| `examples/synthetic_demo/train.jsonl` | Generated-tone records used as the training manifest for software tests. |
| `requirements.txt` | Required Python packages and optional evaluation dependencies. |
| `runs/demo/five_seeds/cer_summary.json` | Cross-seed CER summary for the generated-tone software tests. |
| `runs/demo/five_seeds/cpu_profile.json` | FP32 CPU timing and memory profile for software-test inputs. |
| `runs/demo/five_seeds/cpu_profile_int8.json` | Dynamic INT8 CPU timing and memory profile for software-test inputs. |
| `runs/demo/five_seeds/seed_101/best.pt` | Saved model checkpoint from generated-tone software-test run 101. |
| `runs/demo/five_seeds/seed_101/execution_metadata.json` | Seed, configuration, hardware, and input-source metadata for software-test run 101. |
| `runs/demo/five_seeds/seed_101/run_summary.json` | Epoch count and checkpoint selection summary for software-test run 101. |
| `runs/demo/five_seeds/seed_101/test/evaluation.json` | Calculated evaluation metrics from software-test run 101. |
| `runs/demo/five_seeds/seed_101/test/predictions.jsonl` | Sample-level predictions from software-test run 101. |
| `runs/demo/five_seeds/seed_101/training.jsonl` | Epoch-by-epoch training and validation logs for software-test run 101. |
| `runs/demo/five_seeds/seed_101/vocab.json` | Character vocabulary saved with software-test run 101. |
| `runs/demo/five_seeds/seed_17/best.pt` | Saved model checkpoint from generated-tone software-test run 17. |
| `runs/demo/five_seeds/seed_17/execution_metadata.json` | Seed, configuration, hardware, and input-source metadata for software-test run 17. |
| `runs/demo/five_seeds/seed_17/run_summary.json` | Epoch count and checkpoint selection summary for software-test run 17. |
| `runs/demo/five_seeds/seed_17/test/evaluation.json` | Calculated evaluation metrics from software-test run 17. |
| `runs/demo/five_seeds/seed_17/test/predictions.jsonl` | Sample-level predictions from software-test run 17. |
| `runs/demo/five_seeds/seed_17/training.jsonl` | Epoch-by-epoch training and validation logs for software-test run 17. |
| `runs/demo/five_seeds/seed_17/vocab.json` | Character vocabulary saved with software-test run 17. |
| `runs/demo/five_seeds/seed_29/best.pt` | Saved model checkpoint from generated-tone software-test run 29. |
| `runs/demo/five_seeds/seed_29/execution_metadata.json` | Seed, configuration, hardware, and input-source metadata for software-test run 29. |
| `runs/demo/five_seeds/seed_29/run_summary.json` | Epoch count and checkpoint selection summary for software-test run 29. |
| `runs/demo/five_seeds/seed_29/test/evaluation.json` | Calculated evaluation metrics from software-test run 29. |
| `runs/demo/five_seeds/seed_29/test/predictions.jsonl` | Sample-level predictions from software-test run 29. |
| `runs/demo/five_seeds/seed_29/training.jsonl` | Epoch-by-epoch training and validation logs for software-test run 29. |
| `runs/demo/five_seeds/seed_29/vocab.json` | Character vocabulary saved with software-test run 29. |
| `runs/demo/five_seeds/seed_42/best.pt` | Saved model checkpoint from generated-tone software-test run 42. |
| `runs/demo/five_seeds/seed_42/execution_metadata.json` | Seed, configuration, hardware, and input-source metadata for software-test run 42. |
| `runs/demo/five_seeds/seed_42/run_summary.json` | Epoch count and checkpoint selection summary for software-test run 42. |
| `runs/demo/five_seeds/seed_42/test/evaluation.json` | Calculated evaluation metrics from software-test run 42. |
| `runs/demo/five_seeds/seed_42/test/predictions.jsonl` | Sample-level predictions from software-test run 42. |
| `runs/demo/five_seeds/seed_42/training.jsonl` | Epoch-by-epoch training and validation logs for software-test run 42. |
| `runs/demo/five_seeds/seed_42/vocab.json` | Character vocabulary saved with software-test run 42. |
| `runs/demo/five_seeds/seed_73/best.pt` | Saved model checkpoint from generated-tone software-test run 73. |
| `runs/demo/five_seeds/seed_73/execution_metadata.json` | Seed, configuration, hardware, and input-source metadata for software-test run 73. |
| `runs/demo/five_seeds/seed_73/run_summary.json` | Epoch count and checkpoint selection summary for software-test run 73. |
| `runs/demo/five_seeds/seed_73/test/evaluation.json` | Calculated evaluation metrics from software-test run 73. |
| `runs/demo/five_seeds/seed_73/test/predictions.jsonl` | Sample-level predictions from software-test run 73. |
| `runs/demo/five_seeds/seed_73/training.jsonl` | Epoch-by-epoch training and validation logs for software-test run 73. |
| `runs/demo/five_seeds/seed_73/vocab.json` | Character vocabulary saved with software-test run 73. |
| `runs/demo/five_seeds/tuned_rerank_demo.json` | Validation-selected CTC semantic re-ranking settings for software-test inputs. |
| `runs/demo/five_seeds/wer_summary.json` | Cross-seed WER summary for the generated-tone software tests. |
| `runs/demo/validation.json` | Validation summary generated by local software tests. |
| `scripts/aggregate_seeds.py` | Combines individual run metrics into mean, standard deviation, and confidence intervals. |
| `scripts/compare_seeds.py` | Compares aligned prediction files with a two-level seed/sample bootstrap. |
| `scripts/evaluate.py` | Runs model inference and exports predictions plus evaluation metrics. |
| `scripts/make_demo_fixture.py` | Creates short electronic-tone fixtures for functional testing. |
| `scripts/prepare_corpora.py` | Converts supported speech and text resources into structured manifests. |
| `scripts/pretrain_auxiliary.py` | Runs auxiliary language and poetry next-token pretraining for LSTM modules. |
| `scripts/profile_edge.py` | Profiles single-utterance CPU execution and optional dynamic INT8 quantization. |
| `scripts/run_seeds.py` | Runs training, validation-based re-ranking selection, and evaluation across multiple seeds. |
| `scripts/score_predictions.py` | Calculates transcription, intent, and generation metrics from saved predictions. |
| `scripts/stream_common_voice.py` | Streams a specified Common Voice locale into local audio files and manifests. |
| `scripts/subjective_statistics.py` | Summarizes paired Likert-scale ratings with Friedman and Kendall-W statistics. |
| `scripts/train.py` | Trains and checkpoints the joint speech–semantic model for one seed. |
| `scripts/tune_rerank.py` | Selects a semantic candidate re-ranking coefficient using validation audio. |
| `scripts/validate_data.py` | Checks manifest audio paths, sample identity, source origin, and speaker separation. |
| `src/cultural_speech/__init__.py` | Marks the cultural speech–semantic implementation as an importable Python package. |
| `src/cultural_speech/config.py` | Loads YAML settings and supplies random-seed, device, checksum, and JSON helpers. |
| `src/cultural_speech/dataset.py` | Reads speech manifests and batches acoustic, intent, and generation targets. |
| `src/cultural_speech/decode.py` | Implements CTC prefix beam search and semantic candidate re-ranking. |
| `src/cultural_speech/features.py` | Extracts 39-dimensional MFCC, delta, and delta-delta features from audio. |
| `src/cultural_speech/metrics.py` | Defines transcription, classification, generation, interval, and statistical metrics. |
| `src/cultural_speech/model.py` | Defines the BiLSTM–CTC network, blank-aware bridge, semantic LSTM, and generator. |
| `src/cultural_speech/training.py` | Runs optimizer steps, validation passes, logging, and checkpoint selection. |
| `src/cultural_speech/vocab.py` | Builds and encodes character vocabularies with reserved special tokens. |
| `tests/test_data.py` | Checks the manifest and dataset validation behavior. |
| `tests/test_metrics.py` | Checks metric calculations against small deterministic examples. |
| `tests/test_model.py` | Checks model outputs, losses, and the semantic-to-acoustic gradient path. |
