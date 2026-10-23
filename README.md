# CIC-IDS2017 vs LycoS-IDS2017

A comparison of two network intrusion detection datasets built from the same traffic capture: **CIC-IDS2017** (flows extracted with CICFlowMeter) and **LycoS-IDS2017** (the same PCAP files re-processed with LycoSTand and relabelled). Both are run through an identical Random Forest pipeline so the results can be compared side by side.

## Pipeline

1. Load the CSV files (features cast to `float32` to save memory).
2. Clean: replace `inf` with `NaN`, drop `NaN` rows, drop duplicate rows.
3. Sample: at most 100,000 rows per class (`random_state=42`). This changes the class distribution.
4. Split: stratified 70% train / 30% test (`random_state=42`).
5. Model: Random Forest, 100 trees, no hyperparameter tuning.
6. Metrics: per-class precision, recall and F1, macro F1, confusion matrix.

For LycoS-IDS2017, the identifier columns `flow_id`, `src_addr`, `dst_addr`, `src_port` and `timestamp` are removed before training so the model cannot memorise identities.

## Results (single run)

| Metric | CIC-IDS2017 | LycoS-IDS2017 |
| --- | --- | --- |
| Accuracy | 0.9970 | 0.9977 |
| Macro F1, all own classes | 0.8960 (15 classes) | 0.9134 (14 classes) |
| Macro F1, 14 shared classes | 0.9044 | 0.9134 |

Both datasets separate DoS, DDoS, PortScan and the Patator attacks almost perfectly. The hard classes are the web attacks (XSS, SQL Injection, Brute Force), which are often confused with each other. LycoS-IDS2017 has no Infiltration class.

The difference between the two datasets is small and comes from one run, so it should not be read as a definitive ranking.

### Cross-dataset experiment

A model trained on one dataset and tested on the other, using 68 matched features and the 14 shared classes:

| Experiment | Accuracy | Macro F1 |
| --- | --- | --- |
| CIC → CIC | 0.9969 | 0.9053 |
| LycoS → LycoS | 0.9977 | 0.8923 |
| CIC → LycoS | 0.7731 | 0.8666 |
| LycoS → CIC | 0.4486 | 0.6398 |

Performance drops sharply across datasets, mainly for short-flow attacks (PortScan, DDoS, DoS Hulk, Bot). Because both datasets come from the same PCAP files, this measures robustness to differences in flow extraction, not generalisation to a different network.

## Limitations

- One run and one random split; no repeated runs or cross-validation.
- Very small test classes (Heartbleed, SQL Injection, Infiltration), so their F1 scores are unreliable.
- Random splits can put near-identical flows in both train and test sets, which inflates scores.
- The two datasets differ in size, class set and feature set, so the effects of flow fixes, feature definitions and relabelling cannot be separated.

## Repository layout

```
notebooks/   Python scripts with `# %%` cells
  01_kesif.py              CIC-IDS2017 exploration (shape, labels, NaN/inf, duplicates)
  02_temizlik_ve_model.py  CIC-IDS2017 cleaning + Random Forest
  03_lycos_model.py        LycoS-IDS2017 pipeline + comparison
  04_capraz_deney.py       cross-dataset experiment
results/     classification reports, confusion matrices, comparison tables
data/        not in the repository (see below)
```

## Data

The datasets are several GB and are not included. Download them and place the files as follows:

- CIC-IDS2017 `MachineLearningCSV` (8 CSV files) → `data/cicids2017/MachineLearningCVE/`
  Source: https://www.unb.ca/cic/datasets/ids-2017.html
- LycoS-IDS2017 labelled CSV files → `data/lycos2017/lycos-ids2017/lycos-ids2017/`
  Source: https://lycos-ids.univ-lemans.fr/

## Usage

```
pip install -r requirements.txt
python notebooks/02_temizlik_ve_model.py
python notebooks/03_lycos_model.py
python notebooks/04_capraz_deney.py
```

The scripts also run cell by cell in the VS Code Interactive window. Run the cells from top to bottom; training takes several minutes, and the cross-dataset script can take 15–30 minutes.

## References

- Sharafaldin, Lashkari and Ghorbani (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018.
- Rosay, Cheval, Carlier and Leroux (2022). *Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017.* ICISSP 2022.
- Lanvin et al. *Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes.*
- Liu et al. (2022). *Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018.* IEEE CNS 2022.
