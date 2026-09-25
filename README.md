# Predicting Fair Pricing of Wedding Photography Vendors Using Web-Scraped Vendor Data

**Business Analytics Individual Case Study**

| | |
|---|---|
| **Student** | Konduru Hemesh |
| **Roll No** | CB.SC.U4CSE23724 |
| **Program** | B.Tech Computer Science & Engineering, Section H |
| **Course** | 23CSE452 Business Analytics |
| **Academic Year** | 2025–2026 |

This project builds a two-stage analytical workflow to investigate pricing patterns among wedding photography vendors in India. Public listing data was self-collected from WedMeGood across 14 Indian cities. A hedonic regression benchmark estimates expected prices from observable vendor attributes; vendors are then categorised as priced below, within, or above that benchmark. A classification model predicts these benchmark-relative categories using only pre-booking observable features.

---

## Problem Statement

Wedding photography is a high-value consumer service characterised by wide variation in quoted starting prices and limited public information on what drives those differences. Couples and planners have few tools to assess whether a vendor's listed price is typical, low, or high relative to comparable vendors in the same market.

This project asks: *do observable vendor characteristics — such as market location, customer ratings, review volume, and stated experience — explain differences in listed daily rates, and can those characteristics be used to classify vendors as priced below, within, or above a model-derived benchmark?*

The output categories — **Underpriced**, **Fairly Priced**, and **Overpriced** — are analytical labels relative to the fitted model. They are not objective assessments of vendor quality or value.

---

## Objectives

1. Collect a large, self-created dataset of wedding photography vendor listings from a public marketplace.
2. Clean and validate pricing, rating, review, experience, location, and service attributes.
3. Explore factors associated with differences in listed daily rates.
4. Build a hedonic pricing benchmark using observable vendor characteristics via Ridge Regression.
5. Apply 5-fold out-of-fold cross-validation to produce residual-based benchmark estimates.
6. Use residual quartiles to define benchmark-relative pricing categories.
7. Build a target-leakage-free classifier using only pre-booking observable features.
8. Compare Logistic Regression and Random Forest classification performance.
9. Derive practical insights for customers, vendors, and marketplace platforms.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | WedMeGood public vendor listings (`wedmegood.com`) |
| **Raw records** | 11,340 |
| **Final clean records** | 11,030 |
| **Markets** | 14 Indian cities |
| **Primary price variable** | `price_per_day_inr` (starting daily package rate, INR) |
| **Price range (clean)** | ₹5,000 – ₹10,00,000 |
| **Median price** | ₹55,000 |
| **Mean price** | ₹69,010 |
| **Rating** | Raw listing rating + `is_rated` indicator |
| **Reviews** | `review_count` |
| **Experience** | `experience_years` (explicitly stated in 892 listings; 10,138 unstated) |
| **Location** | City + locality |
| **Collection type** | Self-collected web-scraped public listings |

**This is not a downloaded Kaggle, UCI, or GitHub dataset.** All records were collected directly from publicly accessible vendor pages.

---

## Data Collection

Vendor listings were collected from WedMeGood across 14 Indian metropolitan and urban markets:

- **Markets:** Delhi NCR, Mumbai, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Surat, Indore, Lucknow, Udaipur, Patna
- A custom Python crawler (`src/scraping/production_scraper.py`) was used to collect vendor pages.
- Raw listings were stored as `data/raw/wedding_photographers_raw.csv` (11,340 records, 19 columns) before any cleaning.
- Vendor-level identifiers (`vendor_id`, `vendor_name`) and source metadata were retained for traceability.
- A prototype crawler (`src/scraping/prototype_scraper.py`) was used for initial source feasibility testing.

---

## Data Cleaning & Preprocessing

Cleaning was performed in a waterfall pipeline (see Section 7 of `analysis.ipynb`):

| Step | Records Before | Removed | Records After |
|---|---:|---:|---:|
| Deduplication on `vendor_id` | 11,340 | 160 | 11,180 |
| Remove missing starting prices | 11,180 | 145 | 11,035 |
| Price range filter [₹5k, ₹10L] | 11,035 | 5 | **11,030** |

**Key methodological decisions:**

- **Duplicate handling:** Vendors appearing multiple times were deduplicated on the unique `vendor_id` primary key, retaining the first occurrence.
- **Missing prices:** Records without a stated starting daily rate were removed; no price values were imputed or synthesised.
- **Unrated vendors:** Listings with a rating of 0.0 are retained as a valid market segment. An explicit binary indicator `is_rated` distinguishes rated from unrated vendors (5,990 unrated, 54.3% of clean dataset).
- **Missing locality:** Listings with no stated locality are retained; locality fields are kept as observed without imputation.
- **Experience missingness:** Operating tenure (`experience_years`) is explicitly stated for only 892 of 11,030 records (8.09%). For modelling, the sample median (8.0 years) is used as a neutral fill value (`experience_years_imputed`). A binary flag `experience_stated` records whether the original listing explicitly stated experience. The imputed value is a modelling treatment — it does not imply the vendor has exactly that many years of experience.
- **No artificial price ranges were generated.** No price range formulas (e.g. 0.7x / 1.5x multipliers) were applied.
- **Log transformation:** `ln(price_per_day_inr)` is used as the regression target to address strong right-skew (linear skewness +4.41).

---

## Analytical Methodology

```
Raw Web Data
     ↓
Data Cleaning & Quality Gates
     ↓
Exploratory Data Analysis
     ↓
Hedonic Pricing Benchmark (Ridge Regression)
     ↓
5-Fold Out-of-Fold Residual Calculation
     ↓
Benchmark-Relative Pricing Classes (IQR Thresholds)
     ↓
Target Leakage Audit
     ↓
Classification Models (Logistic Regression / Random Forest)
     ↓
Evaluation, Feature Importance & Sensitivity Analysis
     ↓
Business Insights & Recommendations
```

### Stage 1 — Hedonic Pricing Benchmark

A log-linear hedonic pricing model is used to estimate the expected market price for each vendor given observable characteristics.

- **Dependent variable:** `ln(price_per_day_inr)`
- **Model:** Ridge Regression (`alpha = 1.0`)
- **Validation:** 5-fold shuffled cross-validation (`random_state=42`)
- **Prediction strategy:** Out-of-fold (OOF) predictions, so each vendor's expected price is estimated from folds it was not used to train — eliminating in-sample optimism.
- **Residual:** `oof_log_residual = ln(actual_price) − ln(oof_predicted_price)`
- **Benchmark categories:** Defined by empirical IQR of the OOF residuals:
  - `Underpriced`: residual < Q1 (−0.3363), i.e. actual price is more than ~28.6% below the model's expectation
  - `Fairly Priced`: residual between Q1 and Q3
  - `Overpriced`: residual > Q3 (+0.2973), i.e. actual price is more than ~34.6% above expectation

These are benchmark-relative categories. They do not constitute an objective measure of vendor quality or a universally correct price.

### Stage 2 — Classification

A supervised classifier predicts the benchmark-relative pricing category from features available before or at the point of vendor evaluation.

**Classifier features (13 pre-booking attributes):**

| Feature | Type |
|---|---|
| `rating_imputed` | Numerical |
| `is_rated` | Binary |
| `review_count` | Numerical |
| `experience_years_imputed` | Numerical |
| `experience_stated` | Binary |
| `cities_covered_count` | Numerical |
| `videography_included` | Binary |
| `award_winner` | Binary |
| `handpicked_badge` | Binary |
| `verified_status` | Binary |
| `city` | Categorical |
| `city_tier` | Categorical |
| `package_type` | Categorical |

**Leakage audit:** A programmatic audit confirmed zero pricing-derived variables (actual price, expected price, residuals, or benchmark labels) in the feature matrix `X`.

**Models compared:**

- Multinomial Logistic Regression (baseline)
- Random Forest Classifier (primary)

**Train/test split:** 80% training (8,824 instances) / 20% test (2,206 instances), stratified by pricing class.

---

## Key Results

| Metric | Value |
|---|---:|
| Hedonic benchmark 5-Fold Mean OOF R² | 0.0869 |
| OOF residual mean | −0.0000 |
| OOF residual std deviation | 0.5329 |
| Residual Q1 (Underpriced cutoff) | −0.3363 |
| Residual Q3 (Overpriced cutoff) | +0.2973 |
| Theoretical random baseline | 33.33% |
| Logistic Regression — test accuracy | 39.17% |
| Logistic Regression — Macro F1 | 0.3537 |
| Random Forest — test accuracy | 41.12% |
| Random Forest — Macro F1 | 0.3657 |
| Random Forest — Weighted F1 | 0.4100 |
| Random Forest lift vs. random baseline | +7.79 pp (+23.3% relative) |

The Random Forest achieved higher test accuracy and Macro F1 than the Logistic Regression baseline. Overall predictive performance is modest, which indicates that pre-booking vendor attributes contain some signal for benchmark-relative pricing but do not fully determine it.

---

## Benchmark Classification

Vendors are classified relative to the fitted hedonic benchmark using empirical IQR boundaries of the OOF log residuals:

| Class | Threshold | Count | Share |
|---|---|---:|---:|
| Underpriced | residual < −0.3363 | 2,758 | 25.0% |
| Fairly Priced | −0.3363 ≤ residual ≤ +0.2973 | 5,514 | 50.0% |
| Overpriced | residual > +0.2973 | 2,758 | 25.0% |

Lower residuals indicate listed prices are below what comparable vendors in similar markets tend to charge. Higher residuals indicate the opposite. These categories depend on the model specification and chosen thresholds and should be interpreted as screening indicators, not absolute valuations.

---

## Feature Importance

Random Forest mean decrease in impurity (Gini importance) for the primary model:

| Feature | Importance |
|---|---:|
| Review count | 23.63% |
| Customer rating (imputed) | 15.01% |
| Operating tenure (imputed) | 13.40% |
| Cities covered | 11.53% |
| Market: Mumbai | 3.52% |
| Market: Kolkata | 3.31% |
| Experience stated flag | 3.64% |
| Is rated flag | 3.17% |
| Handpicked badge | 2.24% |
| Award winner | 0.90% |

Feature importance reflects predictive contribution within the fitted model. It does not establish causal relationships between these attributes and vendor pricing decisions.

---

## Sensitivity Analysis

To assess the influence of the continuous imputed experience variable, an ablation test compares:

| Specification | Features | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| Model A | `experience_years_imputed` + `experience_stated` | 41.12% | 0.3657 |
| Model B | `experience_stated` only (no continuous tenure) | 41.61% | 0.3718 |

Accuracy delta: **+0.50 pp**. Macro F1 delta: **+0.0061**.

Removing the continuous imputed experience variable produced only a small change in observed classification performance. This suggests the model's reported performance is not highly sensitive to that particular imputed continuous feature.

---

## Exploratory Findings

**Price distribution:**
- Prices span ₹5,000 to ₹10,00,000 with severe right-skew (skewness +4.41 on linear scale, −0.09 on log scale).
- Median: ₹55,000. Mean: ₹69,010.

**City-level differences:**
- Tier 1 cities and destination hubs show higher median rates. Bangalore, Delhi NCR, and Udaipur show higher median rates than Patna, Indore, and Lucknow.
- Geographic market is a meaningful predictor of listed price.

**Rating patterns:**
- Most rated vendors cluster between 4.8 and 5.0 (strong rating compression on the platform).
- 5,990 vendors (54.3%) are unrated — these are retained as a distinct market segment.

**Review volume:**
- Review count correlates positively with price (ρ = 0.38). The majority of vendors have zero reviews (median = 0).

**Experience reporting:**
- Experience is explicitly stated in only 892 of 11,030 listings (8.09%). This non-disclosure is treated as a structural characteristic of the market rather than a data error.

**Service attributes:**
- Handpicked badge, award status, and verified status are present in a minority of listings and contribute modest but non-zero predictive signal.

---

## Comparison with Published Literature

| Study | Domain | Method | Relevance to This Project |
|---|---|---|---|
| **Carnehl, Stenzel & Schmidt (2024)** *Management Science* DOI: [10.1287/mnsc.2023.4771](https://doi.org/10.1287/mnsc.2023.4771) | Online platform rating systems and dynamic pricing | Econometric dynamic game formulation, regression | Informs the inclusion of rating availability (`is_rated`) and rating tiers to capture platform-salient reputational signals |
| **Namburu, Selvaraj & Varsha (2024)** *Innovations in Systems and Software Engineering* DOI: [10.1007/s11334-022-00465-3](https://doi.org/10.1007/s11334-022-00465-3) | Product and service pricing from scraped listings | Hybrid ML (Random Forest, Gradient Boosting, Linear) | Supports the model architecture choice of comparing a linear baseline against a tree ensemble for pricing classification |
| **Tan, Su, Wu, Cheng & Zheng (2024)** *Sustainability*, 16(15), 6384. DOI: [10.3390/su16156384](https://doi.org/10.3390/su16156384) | Peer-to-peer rental marketplace (Airbnb) pricing | Hedonic model vs. ensemble and deep learning | Parallels the two-stage approach: econometric hedonic baseline for expected prices combined with a classifier for pricing categories |

These studies informed the methodology; this project does not claim to reproduce or validate their findings.

---

## Business Insights

### For Customers
- Use benchmark-relative classification as an initial screening tool when comparing vendors at similar market locations and service tiers.
- Consider rating, review volume, experience disclosure, and service attributes together rather than any single metric.
- A vendor classified as Underpriced relative to comparable listings warrants further evaluation — not automatic selection.

### For Vendors
- The analysis identifies where listed prices sit relative to comparable vendors in the same market, as estimated by observable characteristics.
- Understanding which observable attributes correlate with higher benchmark expectations can inform pricing strategy and profile completeness.
- The analysis is a diagnostic, not a prescription. It does not identify an optimal price.

### For Platforms
- Providing benchmark or comparable-vendor pricing ranges could meaningfully reduce information asymmetry for customers.
- Standardising service and package information (e.g. hours, albums, videography) would improve comparability.
- Encouraging experience and service-attribute disclosure may improve both pricing transparency and model predictability.

These recommendations are derived from the observed data and model results. Causal conclusions are not warranted from observational data alone.

---

## Limitations

- Listed starting prices are publicly stated package rates and may differ from final negotiated contract prices.
- Service quality variables not visible in the listing (portfolio quality, client communication, equipment) are not captured.
- Operating tenure is unstated in 91.91% of profiles; the imputed median is a modelling treatment, not an observed value.
- 54.3% of vendors have no reviews, limiting the utility of review-based signals for a large portion of the dataset.
- Vendor listings were collected from a single platform (WedMeGood); results may not generalise to other platforms or channels.
- Data was collected at a specific point in time and may not reflect current market conditions.
- Observational data does not establish causality between vendor attributes and pricing decisions.
- Overall classification performance is modest (41.12% accuracy against a 33.33% random baseline), indicating substantial unexplained variance.
- Benchmark-relative categories depend on the chosen model specification and residual IQR thresholds.

---

## Repository Structure

```
wedding_photography_price_analysis/
├── README.md
├── analysis.ipynb
├── Case_Study_Report.pdf
├── requirements.txt
└── data/
    ├── raw/
    │   └── wedding_photographers_raw.csv
    └── cleaned/
        └── wedding_photographers_cleaned.csv
```

> **Note:** Development scripts (`src/`), output figures (`outputs/`), intermediate datasets (`data/processed/`, `data/final/`), and scraper logs are maintained locally but excluded from this repository.

---

## Reproducibility

### Environment Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running the Notebook

Open `analysis.ipynb` in Jupyter or VS Code. The notebook is pre-executed with all outputs. To re-execute from scratch:

```powershell
python -m nbconvert --to notebook --execute --inplace analysis.ipynb
```

### Generating the PDF Report

The PDF report was generated using `src/generate_pdf_report.py` (requires ReportLab):

```powershell
python src/generate_pdf_report.py
```

---

## Submission Artifacts

| File | Description |
|---|---|
| `analysis.ipynb` | Complete executed analytical workflow (64 cells, all outputs present) |
| `Case_Study_Report.pdf` | Academic case-study report |
| `data/raw/wedding_photographers_raw.csv` | Raw self-collected dataset (11,340 records) |
| `data/cleaned/wedding_photographers_cleaned.csv` | Cleaned dataset used in analysis (11,030 records) |
| `requirements.txt` | Python package dependencies |

---

## Academic Integrity & Data Provenance

This project uses a self-collected dataset of publicly available vendor listings from WedMeGood. The analysis preserves a clear distinction between observed values, derived variables, and modelling treatments.

- All 11,030 records are authentic public listings.
- No price values, experience values, or review counts were synthesised or fabricated.
- Missing values are handled transparently; imputed variables are clearly flagged with corresponding indicator columns.
- Pricing-derived variables (expected price, OOF residuals, benchmark labels) were programmatically excluded from the classifier feature matrix and confirmed by a leakage audit.
- The dataset is not sourced from Kaggle, UCI, or any pre-existing public repository.

---

## References

1. Carnehl, C., Stenzel, A., & Schmidt, P. (2024). Pricing for the Stars: Dynamic Pricing in the Presence of Rating Systems. *Management Science*. DOI: [10.1287/mnsc.2023.4771](https://doi.org/10.1287/mnsc.2023.4771)
2. Namburu, A., Selvaraj, P., & Varsha, M. (2024). Product pricing solutions using hybrid machine learning algorithm. *Innovations in Systems and Software Engineering*. DOI: [10.1007/s11334-022-00465-3](https://doi.org/10.1007/s11334-022-00465-3)
3. Tan, H., Su, T., Wu, X., Cheng, P., & Zheng, T. (2024). A Sustainable Rental Price Prediction Model Based on Multimodal Input and Deep Learning — Evidence from Airbnb. *Sustainability*, 16(15), 6384. DOI: [10.3390/su16156384](https://doi.org/10.3390/su16156384)
4. Rosen, S. (1974). Hedonic Prices and Implicit Markets: Product Differentiation in Pure Competition. *Journal of Political Economy*, 82(1), 34–55.
5. WedMeGood. (2026). Wedding Photographers Directory Across Indian Cities. Publicly accessible directory. Retrieved from [wedmegood.com](https://www.wedmegood.com)

---

**Author:** Konduru Hemesh  
**Course:** 23CSE452 Business Analytics  
**Academic Year:** 2025–2026  

*This repository is submitted for individual case-study evaluation as part of the B.Tech CSE programme at Amrita Vishwa Vidyapeetham, Coimbatore.*
