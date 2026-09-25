# Predicting Fair Pricing of Wedding Photography Vendors Using Web-Scraped Vendor Data

### Business Analytics Individual Case Study Submission
* **Student Name:** Konduru Hemesh  
* **Roll Number:** CB.SC.U4CSE23724  
* **Program:** B.Tech Computer Science & Engineering, Section H  
* **Course:** 23CSE452 Business Analytics (Semester VI, 2025–2026)  
* **Primary Submission Artifacts:** `analysis.ipynb` | `Case_Study_Report.pdf`

---

## 1. Project Overview

Wedding photography in India is an experiential, high-stakes consumer service characterized by severe price opacity, fragmented regional supply, and information asymmetry. Quoted starting package rates vary from under ₹15,000 to over ₹3,50,000 per day without standardized market benchmarks connecting quoted fees to measurable vendor quality or service deliverables.

This project delivers a comprehensive, academically validated Business Analytics workflow:
1. **Self-Collected Granular Dataset:** Harvested **11,340 raw public vendor listings** across 14 Indian metropolitan markets from WedMeGood (`wedmegood.com`) using a custom multi-threaded crawler, yielding **11,030 clean, non-duplicate records** (strictly exceeding the mandatory $\ge 10,000$ academic threshold).
2. **Methodological Preprocessing:** Zero fabricated price ranges (no 0.7x or 1.5x formulas), zero artificial review-to-experience heuristics, neutral median imputation for operating tenure (8.0 yrs), preservation of unrated vendors (`is_rated = 0`), and preservation of missing localities as `'Unknown'`.
3. **Out-of-Fold (OOF) Hedonic Pricing Benchmark:** Formulates an econometric log-linear regression ($\text{Ridge}, \alpha=1.0$) evaluated via **5-Fold Cross-Validation** to eliminate in-sample optimism ($R^2 = 0.0869$). Empirical Interquartile Range (IQR) residual boundaries ($Q_1 = -0.3363$, $Q_3 = +0.2973$) objectively segment vendors into **Underpriced** (25.0%), **Fairly Priced** (50.0%), and **Overpriced** (25.0%) cohorts.
4. **Leakage-Free Predictive Classifier:** Purges all pricing metrics, residuals, and expected prices from the feature space, predicting pricing classes using strictly pre-booking observable attributes via a **Random Forest Classifier** (41.12% test accuracy and 0.3657 Macro F1 vs. 33.33% random baseline and 39.17% Logistic Regression baseline).
5. **Comparison with Published Literature:** Extensively benchmarked against three recent peer-reviewed studies:
   * **Carnehl, Stenzel, & Schmidt (2024)**, *Management Science* (Dynamic pricing & rating thresholds).
   * **Namburu, Selvaraj, & Varsha (2024)**, *Innovations in Systems and Software Engineering* (Product pricing with hybrid ML).
   * **Tan, Su, Wu, Cheng, & Zheng (2024)**, *Sustainability* (Hedonic pricing vs. ensemble models on Airbnb).
6. **Academic Submission Artifacts:** Includes a fully executed, reproducible implementation notebook ([`analysis.ipynb`](analysis.ipynb)) and an executive 6-page PDF report ([`Case_Study_Report.pdf`](Case_Study_Report.pdf)).

---

## 2. Official Submission File Structure

```text
Business-Analytics-Case-Study/
│
├── README.md                                  # Comprehensive project & reproduction guide
│
├── data/
│   ├── raw/
│   │   └── wedding_photographers_raw.csv      # 11,340 raw crawled listing records
│   │
│   └── cleaned/
│       └── wedding_photographers_cleaned.csv  # 11,030 clean verified records (>= 10,000)
│
├── analysis.ipynb                             # Master implementation notebook with all outputs
│
├── Case_Study_Report.pdf                      # Official 6-page comprehensive academic PDF report
│
├── outputs/                                   # High-resolution figures and evaluation tables
│   ├── figures/                               # Visualizations (Figures 1 to 9)
│   └── tables/                                # Summary metrics and benchmark reports
│
└── src/                                       # Modular Python implementation scripts
    ├── scraping/                              # Web crawlers and source inspection
    ├── preprocessing/                         # Cleaning and quality gate pipelines
    ├── analysis/                              # Exploratory data analysis scripts
    └── modelling/                             # Hedonic benchmark and classifiers
```

---

## 3. Key Analytical Findings & Metrics

| Metric / Parameter | Value | Academic Verification |
| :--- | :--- | :--- |
| **Final Clean Records** | **11,030** | Verified on unique `vendor_id` primary key (Exceeds $\ge 10,000$) |
| **Metropolitan Markets** | **14 Indian Cities** | Delhi NCR, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Surat, Indore, Lucknow, Udaipur, Patna |
| **Directly Observed Price** | **100.0% (₹5,000–₹10,00,000)** | Starting daily package rate; median ₹50,000, mean ₹66,312 (linear skew: +4.41, log skew: -0.09) |
| **Unrated Vendors** | **5,990 (54.3%)** | Preserved raw rating 0.0 with explicit `is_rated = 0` indicator |
| **Operating Tenure** | **892 stated (8.09%)** | Stated experience preserved; unstated (91.91%) imputed to neutral sample median (8.0 yrs) |
| **Hedonic Benchmark** | **5-Fold OOF Ridge ($\alpha=1.0$)** | Mean OOF $R^2 = 0.0869$; centered residual mean $-0.0000$, std dev $0.5329$ |
| **Empirical IQR Bounds** | **$Q_1 = -0.3363$, $Q_3 = +0.2973$** | Underpriced ($> 28.6\%$ below expectation), Fairly Priced (middle 50%), Overpriced ($> 34.6\%$ above) |
| **Leakage Audit** | **0 Leaked Features (PASSED)** | Feature matrix $X$ contains 13 pre-booking features; zero price metrics |
| **Train / Test Partition** | **8,824 Train / 2,206 Test** | Stratified 80/20 train/test split with `random_state=42` |
| **Logistic Regression Baseline** | **39.17% Test Accuracy** | Macro Precision: 36.19%, Macro Recall: 37.58%, Macro F1: 0.3537 |
| **Random Forest Primary Model** | **41.12% Test Accuracy** | Macro Precision: 37.56%, Macro Recall: 38.31%, Macro F1: 0.3657, Weighted F1: 0.4100 |
| **Lift vs. Random Baseline** | **+7.79% absolute (+23.4% relative)** | Random guessing: 33.33%; Random Forest: 41.12% on real-world observational data |
| **Sensitivity Analysis** | **Model A (41.12%) vs Model B (41.61%)** | Proves continuous imputation does not distort outcomes; disclosure flag captures primary signal |
| **Final Automated Audit** | **15 / 15 Criteria PASSED** | Top-to-bottom reproducibility verified with `FINAL NOTEBOOK AUDIT: PASS` |

---

## 4. Comparison with Published Studies

| Published Study | Domain & Dataset | Methodology | Key Finding | Comparison to Our Study |
| :--- | :--- | :--- | :--- | :--- |
| **Carnehl, Stenzel, & Schmidt (2024)**<br>*Management Science*<br>DOI: 10.1287/mnsc.2023.4771 | Online platform rating systems; empirical transaction records | Dynamic game formulation & econometric regression | Sellers adjust pricing around rating cutoffs; reputational premiums materialize once passing platform-salient thresholds. | Informs our inclusion of rating availability indicators (`is_rated`) and rating tiers to capture reputational premiums. |
| **Namburu, Selvaraj, & Varsha (2024)**<br>*Innovations in Systems & Software Eng.*<br>DOI: 10.1007/s11334-022-00465-3 | Product & service pricing; scraped listings & reviews | Hybrid machine learning (Random Forest, Gradient Boosting, Linear) | Non-linear ensemble tree architectures substantially outperform linear models by learning cross-feature interactions. | Supports our model architecture comparing Multinomial Logistic Regression against Random Forest for pricing classification. |
| **Tan, Su, Wu, Cheng, & Zheng (2024)**<br>*Sustainability*, 16(15), 6384<br>DOI: 10.3390/su16156384 | Peer-to-peer rental marketplace (Airbnb listings) | Hedonic pricing model compared with ensemble & deep neural nets | Hedonic models establish broad market equilibrium, but ensemble algorithms better predict non-linear amenity premiums. | Parallels our two-stage approach: econometric hedonic Ridge regression for expected prices combined with Random Forest classification. |

---

## 5. Reproduction & Execution Guide

### Step 1: Environment Setup
Ensure Python 3.11+ is installed. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Executing the Master Implementation Notebook (`analysis.ipynb`)
The primary analytical submission artifact is [`analysis.ipynb`](analysis.ipynb). It is pre-executed with all 66 cells, 10 embedded plots, tables, and audit gates. To re-execute from scratch:
```powershell
python -m nbconvert --to notebook --execute --inplace analysis.ipynb
```

### Step 3: Generating the Official 6-Page PDF Report (`Case_Study_Report.pdf`)
The formal academic PDF report is generated programmatically using ReportLab:
```powershell
python src/generate_pdf_report.py
```
*Verification:* Checks that the PDF page count is strictly within the 5–7 page range (generates exactly 6 pages).

---

## 6. Academic Integrity & Verification Statement

This Business Analytics individual case study represents original, self-collected work by **Konduru Hemesh (CB.SC.U4CSE23724)**:
* **No Synthetic Records:** 100% of vendor records are authentic public listings harvested from WedMeGood.
* **No Manufactured Attributes:** Zero artificial price ranges, zero review-to-experience bucket heuristics, and no fabricated localities.
* **Zero Target Leakage:** No pricing variables, expected prices, or residuals enter the classifier feature matrix $X$.
* **Fully Audited:** Passes all 15 automated compliance criteria with zero exceptions.
