# Spam e-mail classification with five classical ML models — ML Assignment 2 (M.Tech AIML, BITS WILP)

**Live app:** https://mlassignment2bits.streamlit.app/  
**Repository:** https://github.com/RohitNishanth/ML_Assignment_2  
**GitHub profile:** https://github.com/RohitNishanth

## a. Problem statement
Build a classifier that decides whether an incoming e-mail is **spam** or **not spam** from simple
word, character and capitalisation statistics of the message. Five classical models are trained on
the same data and compared on six metrics, and the best one is exposed through a Streamlit app that
evaluates the saved models on uploaded test data.

The two errors do not cost the same. A **false positive** (a genuine e-mail sent to the spam folder)
can mean a missed invoice or interview call, while a **false negative** (spam reaching the inbox) is
only an annoyance. Precision on the spam class therefore matters at least as much as recall, which
is why the final choice below does not rest on accuracy alone.

## b. Dataset description
| Item | Value |
|---|---|
| Name and source | Spambase, UCI Machine Learning Repository (dataset id 94), https://archive.ics.uci.edu/dataset/94/spambase — donated by Hewlett-Packard Labs, 1999 |
| Raw instances | 4,601 e-mails (1,813 spam, 2,788 not spam) |
| Instances used | 4,210 after removing 391 exact duplicate rows (prevents the same e-mail appearing in both train and test) |
| Features | 57, all numeric (requirement: ≥ 12) |
| Target | `email_class` — binary: `spam` (positive class) / `not_spam` |
| Class balance (after de-duplication) | 2,531 not spam (60.1%), 1,679 spam (39.9%) |
| Missing values | None |
| Train / test split | 80 / 20, stratified, random seed 2611 → 3,368 train rows, 842 test rows (506 not spam, 336 spam). The test split is saved as `test_data.csv`. |

Feature groups:
- **48 `word_freq_<word>`** — percentage of words in the e-mail equal to that word (e.g. `free`, `money`, `remove`, `000`, `george`, `hp`, `meeting`).
- **6 `char_freq_<char>`** — percentage of characters that are `;` `(` `[` `!` `$` `#`.
- **3 `capital_run_length_*`** — average length, longest length and total number of runs of capital letters.

The features are heavily right-skewed (most e-mails contain none of a given word), and some words
such as `george`, `hp` and `650` refer to the original donor's own name, employer and area code. They
are strong "not spam" signals in this dataset but would not transfer to another mailbox.

## c. GitHub repository
https://github.com/RohitNishanth/ML_Assignment_2

```
project-folder/
├── app.py               # Streamlit app: loads saved models, never retrains
├── train_models.py      # trains + evaluates the 5 models, saves all artifacts
├── metrics_utils.py     # metric code shared by training and the app
├── requirements.txt     # pinned versions (must match the training environment)
├── README.md
├── test_data.csv        # held-out 20% test split (57 features + email_class)
├── data/spambase.csv    # full dataset with named columns
├── .streamlit/config.toml
└── model/
    ├── logistic_regression.joblib, decision_tree.joblib, knn.joblib,
    │   naive_bayes.joblib, svm.joblib      # full pipelines (preprocessing + model)
    ├── metadata.json                       # feature list, class labels, versions
    ├── metrics.csv, metrics_table.md       # test-set results
    └── train_vs_test.csv                   # train vs test accuracy and tuned parameters
```

Reproduce:
```
pip install -r requirements.txt
python train_models.py --data data/spambase.csv --target email_class --positive spam --seed 2611 --tune
streamlit run app.py
```

## d. Models used and evaluation
Each model is a scikit-learn `Pipeline`: median imputation and standard scaling of all features,
followed by the classifier. Hyper-parameters were chosen by 5-fold stratified grid search on the
**training split only** (weighted F1 score); the test split was used once, for the final numbers below.

| Model | Tuned setting |
|---|---|
| Logistic Regression | C = 1 |
| Decision Tree | max_depth = 10, min_samples_leaf = 1 |
| KNN | k = 11, distance-weighted |
| Naive Bayes (Gaussian) | var_smoothing = 1e-5 |
| SVM (RBF kernel) | C = 5, gamma = "scale", Platt-calibrated for probabilities |

Precision, Recall and F1 are for the positive class `spam`. AUC is the ROC-AUC computed from
predicted spam probabilities. All metrics are on the 842-row held-out test set.

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9157 | 0.9648 | 0.9180 | 0.8661 | 0.8913 | 0.8234 |
| Decision Tree | 0.9062 | 0.8988 | 0.8859 | 0.8780 | 0.8819 | 0.8041 |
| KNN | 0.9014 | 0.9506 | 0.9175 | 0.8274 | 0.8701 | 0.7938 |
| Naive Bayes | 0.8361 | 0.9395 | 0.7271 | 0.9435 | 0.8212 | 0.6941 |
| Support Vector Machine | 0.9192 | 0.9694 | 0.9187 | 0.8750 | 0.8963 | 0.8309 |

Confusion counts on the test set (506 not spam, 336 spam):

| Model | TN | FP (good mail flagged) | FN (spam missed) | TP | Train acc. | Test acc. |
|---|---|---|---|---|---|---|
| Logistic Regression | 480 | 26 | 45 | 291 | 0.928 | 0.916 |
| Decision Tree | 468 | 38 | 41 | 295 | 0.970 | 0.906 |
| KNN | 481 | 25 | 58 | 278 | 0.999 | 0.901 |
| Naive Bayes | 387 | 119 | 19 | 317 | 0.826 | 0.836 |
| Support Vector Machine | 480 | 26 | 42 | 294 | 0.961 | 0.919 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best on accuracy (0.916), AUC (0.965), precision, F1 and MCC (0.823) and only 1.3 points between train and test accuracy, so it generalises almost perfectly. A linear model does this well because spam here is largely additive evidence: high frequencies of `000`, `$`, `remove` and long capital runs push towards spam, while `george`, `hp`, `edu` and `meeting` push away. Its weakness is recall (0.866): 45 spam e-mails got through, typically ones with few of these "loud" words. |
| Decision Tree | Even after limiting depth to 10, it overfits more than the linear models (train 0.970 vs test 0.906). It has the **lowest AUC (0.899)**, because a tree gives coarse, leaf-level probabilities that rank e-mails poorly. Its recall (0.878) is slightly higher than LR's, but it pays with 38 false positives — 46% more genuine e-mails flagged than LR or SVM. It is the most interpretable model, but the least reliable here. |
| KNN | Shows the **largest generalisation gap** (train 0.999 vs test 0.901). That figure is largely an artefact of distance weighting, which makes every training point its own nearest neighbour, but the test numbers are still the weakest of the four strong models. It has good precision (0.918, 25 FP) but the **lowest recall (0.827, 58 spam missed)**. With 57 sparse, skewed dimensions, Euclidean distance is dominated by the few non-zero features, so many spam e-mails sit closer to legitimate ones. It is also the heaviest model to ship (it stores the training set: about 390 KB) and the slowest to predict. |
| Naive Bayes | The odd one out. It has the **best recall (0.944, only 19 spam missed)** but by far the worst precision (0.727): **119 of 506 genuine e-mails (23.5%) were marked as spam**. This drags accuracy to 0.836 and MCC to 0.694. Its two assumptions both fail here: the features are not independent (e.g. `hp`/`hpl`/`650`, and `857`/`415`/`telnet` occur together, so the same evidence is counted several times), and they are not Gaussian (mostly zeros with long tails). Its AUC is still a respectable 0.940, so the probability ranking is reasonable; with a much higher decision threshold it would be usable, but at the default 0.5 threshold it is unacceptable for a mail filter. |
| Support Vector Machine | **Best on five of the six metrics**: accuracy 0.919, AUC 0.969, precision 0.919, F1 0.896 and MCC 0.831. Only recall (0.875) is beaten, by Naive Bayes and narrowly by the Decision Tree. Its false positives equal LR's (26), and it catches 3 more spam e-mails. The RBF kernel captures some non-linear interactions that LR cannot, but the margin over LR is small (about 0.4 points accuracy, 0.008 MCC), roughly 3 e-mails out of 842, which is within the noise of a single split. Costs: the slowest training time (about 13 s with tuning versus under 1 s for LR), a moderate train/test gap (0.961 vs 0.919), and its probabilities need a separate calibration step. |
| **Overall winner for this dataset** | **Support Vector Machine** — it has the highest MCC (0.831), the most informative single metric for this 60/40 imbalanced problem, the highest AUC (0.969), and has close to the fewest false positives (26; only KNN has fewer, at 25, but it misses 16 more spam e-mails), which is the costly error in spam filtering. **Logistic Regression is a near-tie** and would be the practical choice in production: it is faster, smaller, more interpretable and generalises more steadily. Naive Bayes ranks last despite the best recall, because flagging nearly a quarter of real e-mails as spam outweighs catching more spam. |

**Limitations:** the results come from a single 80/20 split, so differences of less than about 1 point
between LR and SVM should not be over-interpreted. The vocabulary features are specific to one
1990s mailbox at HP, so these models would not transfer directly to modern e-mail.

## Streamlit app features
- **Upload test data (CSV)** with the 57 features and the `email_class` column; the held-out `test_data.csv` can be downloaded from the sidebar.
- **Model selection dropdown** for the five saved models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix heat-map and classification report.**
- **Comparison tab:** all five models on the uploaded file, with a metric bar chart and a CSV download.
- The models are loaded from `model/` and only used for prediction; nothing is retrained on uploaded data. Missing columns or unknown labels produce a clear error message.
