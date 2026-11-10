# Federated Human Activity Recognition on WISDM Actitracker

**AI for Trustworthy Decision Making — Stage 1: Baseline Development and Minimal Federated Setup**

**Team:** Yannick Koning, Mihai-Dragos Toma

A phone's accelerometer is enough to recognise what its owner is doing (walking, sitting, etc.), but this sensor data is personal. In **federated learning (FL)** each client trains on its own data and only sends model weights to a server, which averages them (**FedAvg**), so the raw data never leaves the client. We compare FedAvg against ordinary **centralised** training (all data in one place) on the **WISDM Actitracker** dataset, using the same model, data splits and test users for both.

## Research questions

- **RQ1.** How close does FedAvg with 5 clients get to centralised training on the same data?
- **RQ2.** Do more training data or more features help, centralised and federated?
- **RQ3.** How well do the models work for **new users**, and for every client equally?

## Contents

| File | Content |
|---|---|
| `notebooks/AITDM_Stage1_A_Data_Centralised.ipynb` | Dataset, preprocessing, 4 data variants, client splits, centralised MLP and Random Forest baselines |
| `notebooks/AITDM_Stage1_B_Federated.ipynb` | FedAvg on the same splits, IID reference, FedProx and balanced loss, comparison with A and with the literature, discussion, limitations, Stage 2 plan |
| `notebooks/AITDM_Stage1_C_Better_Features.ipynb` | 71 hand-crafted features on the same splits, centralised and FedAvg |
| `presentation/Presentation_Stage1.pdf` | Stage 1 slides |

The notebooks are saved with all outputs, so they can be read without running them. Read them in the order A → B → C.

## Dataset

**WISDM Actitracker v2.0** (WISDM Lab, Fordham University): phone accelerometer (x, y, z) at 20 Hz, recorded in daily life by 225 people, 6 activities (Walking, Jogging, Stairs, Sitting, Standing, LyingDown), ≈ 2.98 million readings.

**Preprocessing:** we drop broken lines and impossible values (> 80 m/s²), then cut the data into windows of 200 readings (10 s). A window never mixes users, activities or recording gaps over 1 s. This gives 13,381 windows from 223 users. **56 users (20 % of windows) are held out completely as the test set**, so every test result measures performance on new people.

## Setup

| | |
|---|---|
| Clients | 5, split by person: each user belongs to exactly one client (non-IID); 15 % of each client's windows are kept for validation |
| Model | MLP: features → 256 → 128 → 6 (ReLU, dropout), Adam, learning rate 0.001, batch size 32 |
| Centralised | the 5 clients' training data pooled, 30 epochs |
| Federated | FedAvg (weighted by client size), 10 rounds × 3 local epochs, Flower simulation |
| Seeds | 1, 2, 3 (results reported as mean ± std) |
| Metrics | accuracy, balanced accuracy, macro-F1, worst-client accuracy |

## Key results (test set = 56 unseen users, mean of 3 seeds)

| Model | Features | Test acc. | Test bal. acc. | Test macro-F1 | Worst client (val) |
|---|---|---|---|---|---|
| Centralised MLP | rich_overlap (32) | 0.801 | 0.678 | 0.666 | 0.920 |
| FedAvg, 5 clients by person | rich_overlap (32) | 0.797 | 0.605 | 0.584 | 0.550 |
| FedAvg, IID clients | basic (12) | 0.796 | 0.653 | 0.624 | 0.842 |
| Centralised MLP | hand-crafted (71) | 0.828 | 0.729 | 0.700 | 0.923 |
| FedAvg, 5 clients by person | hand-crafted (71) | 0.829 | 0.603 | 0.589 | 0.802 |

**Findings**

- FedAvg matches centralised **accuracy** on new users, but loses on **rare activities** (LyingDown, Stairs) and on **one client** (client 0: 0.92 → 0.55).
- The IID reference shows the cost comes from **non-IID data**, not from federated training itself.
- Better hand-crafted features help most: the centralised model rises to 0.73 balanced accuracy, and the weakest FedAvg client to 0.80.
- FedProx and a balanced loss each improve one metric at the expense of another, so no single setting fixes everything.

## How to reproduce

The notebooks run in Google Colab on a CPU runtime. They clone the course repository [AITDM_FLOWER_PROJECT_TUTORIAL](https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL) and reuse its scripts (`split_data.py`, `centralised.py`, `run_seeds.py` with the Flower app, `summarize.py`, `error_analysis.py`). They download the dataset themselves.

1. Open notebook **A** in Colab and run all cells. It downloads the data, builds the client splits and trains the centralised models.
2. Run notebook **B** (federated experiments), then notebook **C** (hand-crafted features). Both use the splits from A.

Data, splits and results are written to `MyDrive/AITDM_WISDM_stage1/` in the Google Drive of the account running the notebooks.

## References

1. Kwapisz, Weiss & Moore (2011). *Activity Recognition using Cell Phone Accelerometers.* SIGKDD Explorations 12(2).
2. Weiss & Lockhart (2012). *The Impact of Personalization on Smartphone-Based Activity Recognition.* AAAI Workshop on Activity Context Representation.
3. Alsheikh et al. (2016). *Deep Activity Recognition Models with Triaxial Accelerometers.* AAAI Workshops (arXiv:1511.04664).
4. Peppas et al. (2020). *Real-Time Physical Activity Recognition on Smart Mobile Devices Using Convolutional Neural Networks.* Applied Sciences 10(23), 8482.
5. McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
6. Beutel et al. (2020). *Flower: A Friendly Federated Learning Research Framework.* arXiv:2007.14390.
