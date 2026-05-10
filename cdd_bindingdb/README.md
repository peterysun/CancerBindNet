# 🧬 CancerBindNet — AI-Powered Cancer Drug Discovery

> Predicting small molecule binding affinity to cancer-relevant protein targets using deep learning and cheminformatics.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023.9%2B-green)](https://rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

CancerBindNet is a machine learning pipeline that predicts whether a drug candidate molecule will bind to — and inhibit — cancer cell receptors. Given any molecule as a SMILES string, the model outputs a predicted IC50 value indicating binding potency.

Trained on **3.1 million experimental measurements** from [BindingDB](https://www.bindingdb.org), covering 10 clinically relevant cancer targets including EGFR (overexpressed in glioblastoma), IDH1/2 (glioma), BRAF, PARP1, and more.

### Validation

Testing against FDA-approved drugs confirms the model captures real binding relationships:

| Molecule | Target | Predicted IC50 | Known IC50 | Result |
|---|---|---|---|---|
| Gefitinib | EGFR | 2.2 nM | ~5 nM | ✅ Strong binder |
| Erlotinib | EGFR | 8.0 nM | ~2 nM | ✅ Strong binder |
| Aspirin | EGFR | ~10,000 nM | No binding | ✅ Correctly rejected |

---

## How It Works

```
Drug molecule (SMILES string)
        ↓
Morgan Fingerprint (2048-bit)     ← RDKit cheminformatics
+ Molecular Descriptors (6-dim)   ← MW, LogP, HBD, HBA, TPSA, RotBonds
        ↓
Neural Network (4 hidden layers)  ← PyTorch
+ Random Forest (200 trees)       ← scikit-learn baseline
        ↓
Predicted pIC50 → IC50 in nM      ← binding affinity
```

### Why Morgan Fingerprints?

Morgan fingerprints encode the local chemical environment of each atom up to a defined radius. Each of the 2048 bits represents presence or absence of a specific chemical substructure. The model learns which substructures correlate with strong binding to each cancer target.

---

## Cancer Targets

| Target | Cancer Type | Role |
|---|---|---|
| **EGFR** | Glioblastoma, Lung | Growth signaling — mutated/overexpressed |
| **IDH1/2** | Glioma | Metabolism — driver mutation |
| **VEGFR** | Multiple | Angiogenesis — tumor blood supply |
| **BRAF** | Melanoma, Colorectal | MAPK signaling |
| **PARP1** | Breast, Ovarian | DNA repair — synthetic lethality |
| **CDK2** | Multiple | Cell cycle control |
| **ABL1** | Leukemia (CML) | Tyrosine kinase — Gleevec target |
| **ALK** | Lung (NSCLC) | Fusion kinase |
| **MET** | Multiple | Growth factor receptor |

---

## Project Structure

```
CancerBindNet/
├── data/
│   └── prepare_bindingdb.py    # Parse & filter BindingDB (handles .zip natively)
├── models/
│   └── binding_model.py        # Neural network + Random Forest architectures
├── utils/
│   ├── molecule_utils.py       # SMILES → fingerprints, Lipinski filter
│   └── metrics.py              # RMSE, Pearson R, Spearman ρ, ROC-AUC, Enrichment Factor
├── scripts/
│   ├── train.py                # Training pipeline with RF baseline comparison
│   └── predict.py              # Score new molecules, rank by predicted potency
├── notebooks/
│   └── CancerBindNet_Colab.ipynb  # Ready-to-run Google Colab notebook
├── requirements.txt
└── README.md
```

---

## Quick Start

### Local

```bash
git clone https://github.com/YOUR_USERNAME/CancerBindNet.git
cd CancerBindNet
pip install -r requirements.txt

# Download BindingDB_All_202605_tsv.zip from bindingdb.org (free)
python data/prepare_bindingdb.py --input path/to/BindingDB_All_202605_tsv.zip

python scripts/train.py --target EGFR --epochs 50
python scripts/predict.py --smiles "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
```

### Google Colab (no GPU required locally)

Open `notebooks/CancerBindNet_Colab.ipynb` in Google Colab for a fully guided walkthrough with free GPU access.

---

## Model Architecture

### Neural Network
```
Input (2054-dim fingerprint + descriptors)
    → Linear(2054, 1024) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(1024, 512)  → BatchNorm → ReLU → Dropout(0.3)
    → Linear(512, 256)   → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256, 128)   → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128, 1)     → pIC50
```

**2.8M parameters** | Trained with Adam optimizer | MSE loss | LR scheduling on plateau

### Random Forest Baseline
- 200 estimators, `max_features='sqrt'`
- Trained on identical feature vectors
- Reported alongside NN for direct comparison

---

## Evaluation Metrics

Beyond standard ML metrics, we report drug-discovery-specific metrics:

| Metric | Description |
|---|---|
| **RMSE / MAE** | Regression error in pIC50 units |
| **Pearson R** | Linear correlation of predictions vs truth |
| **Spearman ρ** | Rank correlation — most important for virtual screening |
| **ROC-AUC** | Active/inactive classification (threshold: IC50 < 1000 nM) |
| **Enrichment Factor** | How many more actives found vs random at top 1% / 5% |

---

## Usage Examples

**Score a single molecule:**
```bash
python scripts/predict.py --smiles "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1"
```

**Score a library of molecules:**
```bash
python scripts/predict.py --smiles-file my_compounds.txt
```

**Train on a specific target:**
```bash
python scripts/train.py --target IDH1 --epochs 100
```

**Train on all cancer targets combined:**
```bash
python scripts/train.py --epochs 50
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 2.0 | Neural network training |
| RDKit | ≥ 2023.9 | Cheminformatics, fingerprints |
| scikit-learn | ≥ 1.3 | Random Forest, train/test split |
| pandas | ≥ 2.0 | Data processing |
| scipy | ≥ 1.11 | Spearman correlation |
| tqdm | ≥ 4.65 | Progress bars |

---

## Background & Motivation

Cancer drug discovery traditionally requires years of wet-lab screening to find molecules that inhibit disease-relevant proteins. Computational approaches using ML can pre-screen millions of candidates in silico, dramatically reducing the time and cost to identify leads.

This project focuses specifically on **brain tumor targets** (EGFR, IDH1/2) connecting to prior work in MRI-based tumor segmentation, with the goal of building an end-to-end pipeline from imaging biomarkers to drug candidate prioritization.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

- Gilson et al. *BindingDB in 2015: A public database for medicinal chemistry.* Nucleic Acids Research, 2016.
- Rogers & Hahn. *Extended-Connectivity Fingerprints.* J. Chem. Inf. Model., 2010.
- Lipinski et al. *Experimental and computational approaches to estimate solubility and permeability.* Adv. Drug Deliv. Rev., 1997.
