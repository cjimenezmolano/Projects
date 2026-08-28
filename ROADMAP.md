# 90-Day ML Engineer Roadmap (local only — not tracked by git)

Started 2026-08-26. Goal: go from "wrote linear regression by hand" to a working
portfolio covering classical ML, deep learning, and the engineering half of ML
engineering (APIs, containers, CI, deployment).

Cloud automation was attempted but blocked (GitHub not connected for cloud-agent
use), so assignments are generated on request instead: ask Claude for "today's
assignment" (or "day N") in a chat, and it checks this file + the repo to build
the next one. That also means it can look at what was actually turned in before
generating the next day, instead of firing blindly on a timer.

## How curriculum drops work

- Claude figures out the next day by checking which `curriculum/day-XXX-*/`
  folders already exist in the repo (highest existing day + 1; day 1 if none
  exist), then looks up that day's topic in the table below.
- `curriculum/day-XXX-<slug>/README.md` — that day's assignment, as
  **structure only**: objective, ordered steps (numbered/bulleted so they can
  be referenced later), explicit acceptance criteria, any constraints (e.g.
  days 1-14 are pure Python only), and a note on why it matters.
  - Each step describes WHAT to do and WHY, specific enough to know exactly
    what function/technique to research and implement — but never names
    exact syntax, library calls, function signatures, or parameters unless
    naming the concept requires it (e.g. "use one-hot encoding" is fine;
    `OneHotEncoder(handle_unknown='ignore')` is not).
  - No code. No pseudocode. No code skeletons or blanks to fill in. No
    `## START CODE HERE ##` markers. No starter `.py` file at all, regardless
    of how well the exercise would fit one.
  - No intros, disclaimers, or motivational wrap-up text in the README —
    just the structure.
  - After generating the README, stop. Do not proactively offer to write
    code, build files, or scaffold the project — wait for the day to be
    implemented and brought back for review.
- A **Hints** section in the README: 2-4 short pointers for when you're stuck
  — the relevant formula or concept to recall, the shape of the approach, a
  common gotcha for that exercise. Hints nudge, they don't solve: no working
  code, no pseudocode, no filled-in blanks. If a hint alone isn't enough, ask
  Claude directly in chat rather than expecting the README to spell out more
  — it can gauge how much to reveal based on where you're actually stuck,
  which a static hint can't.
- One new day-folder per request, in order. You write the entire
  implementation yourself, from a blank file, and commit it yourself.
  Existing `curriculum/day-*/` folders are never touched or overwritten —
  only a new one gets added. If a day gets skipped, the next request just
  picks up the next undone day.

## Phase 1 (Days 1-14): Pure-Python ML fundamentals — no libraries

| Day | Topic |
|-----|-------|
| 1 | Logistic regression from scratch (sigmoid, cross-entropy, gradient descent) |
| 2 | Train/test split + accuracy/precision/recall computed by hand |
| 3 | Multi-class logistic regression (softmax) from scratch |
| 4 | k-Nearest Neighbors from scratch |
| 5 | Decision tree (single-feature splits, Gini impurity) from scratch |
| 6 | k-Means clustering from scratch |
| 7 | Single-neuron perceptron from scratch |
| 8 | Tiny 1-hidden-layer neural net — manual forward pass on XOR |
| 9 | Manual backpropagation for that network (no autograd) |
| 10 | Mini-batch vs. full-batch gradient descent from scratch |
| 11 | L2-regularized (ridge) regression from scratch |
| 12 | k-fold cross-validation from scratch |
| 13 | Confusion matrix, ROC/AUC computed by hand |
| 14 | Refactor days 1-13 into a small reusable `toyml` package |

## Phase 2 (Days 15-30): numpy, pandas, scikit-learn on real data

| Day | Topic |
|-----|-------|
| 15 | Vectorize Day-1 linear/logistic regression with numpy; benchmark vs. pure Python |
| 16 | Vectorize the Day-8/9 neural net with numpy |
| 17 | pandas basics — load and explore a real CSV dataset |
| 18 | EDA: missing values, distributions, correlations |
| 19 | Feature engineering: encoding categoricals, scaling |
| 20 | scikit-learn LinearRegression/LogisticRegression — compare to your from-scratch results |
| 21 | scikit-learn train/test split, cross_val_score, GridSearchCV |
| 22 | scikit-learn decision trees & random forests |
| 23 | scikit-learn SVMs |
| 24 | scikit-learn Pipelines + ColumnTransformer |
| 25 | Imbalanced data: class weights, resampling |
| 26 | Evaluation deep dive: precision/recall tradeoffs, calibration |
| 27 | Save/load models (joblib) + a small inference script |
| 28 | Mini project: end-to-end classic pipeline on a Kaggle dataset (start) |
| 29 | Mini project: continue |
| 30 | Mini project: write up findings, polish README |

## Phase 3 (Days 31-45): Deep learning with PyTorch

| Day | Topic |
|-----|-------|
| 31 | PyTorch basics: tensors, autograd |
| 32 | Reimplement Day-1 regression in PyTorch; compare to hand-rolled version |
| 33 | Reimplement the Day-8/9 neural net as an `nn.Module` |
| 34 | MLP classifier on a tabular dataset |
| 35 | Training loop structure: optimizer, loss, epochs, validation |
| 36 | Regularization in DL: dropout, weight decay |
| 37 | LR schedules and early stopping |
| 38 | Load MNIST via torchvision |
| 39 | MLP for MNIST digit classification |
| 40 | CNN concepts: convolution, pooling |
| 41 | Small CNN for MNIST; compare to the MLP |
| 42 | Data augmentation basics |
| 43 | Save/load PyTorch checkpoints |
| 44 | Mini project: image classifier on a harder dataset (e.g. CIFAR-10 subset) (start) |
| 45 | Mini project: continue, write up learnings |

## Phase 4 (Days 46-60): NLP and sequence models

| Day | Topic |
|-----|-------|
| 46 | Text preprocessing: tokenization, vocab building from scratch |
| 47 | Bag-of-words + scikit-learn text classifier |
| 48 | Pretrained word embeddings (e.g. GloVe) |
| 49 | RNN concepts + a small RNN in PyTorch |
| 50 | LSTM/GRU overview and example |
| 51 | Attention mechanism: concept + a tiny manual example |
| 52 | Transformer architecture overview (read + diagram) |
| 53 | Pretrained transformer via Hugging Face for sentiment classification |
| 54 | Fine-tune a small pretrained model on a custom text dataset |
| 55 | NLP evaluation: accuracy, F1, confusion matrix for text |
| 56 | Mini project: fine-tuned text classifier end to end (start) |
| 57 | Mini project: continue |
| 58 | Embeddings for retrieval: semantic search basics |
| 59 | Build a tiny semantic search demo |
| 60 | Polish NLP project, write README |

## Phase 5 (Days 61-75): The "engineer" half — MLOps

| Day | Topic |
|-----|-------|
| 61 | Wrap a trained model in a FastAPI inference endpoint |
| 62 | Unit tests for ML code with pytest |
| 63 | Dockerize the FastAPI service |
| 64 | Request/response logging + input validation |
| 65 | Experiment tracking with MLflow (or W&B) |
| 66 | Data versioning basics (DVC or checksumming) |
| 67 | Config management for experiments (yaml-based configs) |
| 68 | CI basics: GitHub Actions running tests on push |
| 69 | Model monitoring concepts: drift detection basics |
| 70 | Batch vs. online inference — implement one |
| 71 | Deploy the Docker container to a free-tier host |
| 72 | Add lint + test CI step across the repo |
| 73 | Write a proper README for the API project |
| 74 | Mini project: "productionized" version of an earlier model (API+Docker+tests+CI) (start) |
| 75 | Mini project: continue, polish |

## Phase 6 (Days 76-90): Capstone, portfolio, interview prep

| Day | Topic |
|-----|-------|
| 76 | Capstone: pick your own problem + dataset |
| 77 | Capstone: EDA |
| 78 | Capstone: feature engineering |
| 79 | Capstone: baseline model |
| 80 | Capstone: iterate and tune |
| 81 | Capstone: evaluation and error analysis |
| 82 | Capstone: API wrapper |
| 83 | Capstone: Dockerize + tests |
| 84 | Capstone: deploy |
| 85 | Capstone: README + architecture diagram |
| 86 | Join a Kaggle competition, submit a first baseline |
| 87 | Iterate on the Kaggle submission |
| 88 | Polish GitHub profile: pin repos, top-level README of the 90-day journey |
| 89 | Write a summary mapping projects → skills demonstrated (portfolio/resume prep) |
| 90 | Retrospective: what you learned, what to study next |
