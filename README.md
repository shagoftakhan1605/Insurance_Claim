# Insurance Claims Analysis and Fraud Detection

Exploratory analysis of 1,000 motor insurance claims and a Random Forest model that predicts whether a claim was reported as fraudulent. All code is in [`Insurance_Claims.ipynb`](Insurance_Claims.ipynb); the data are in `insurance_claims.csv`.

---

## Dataset

* 1,000 claims, 39 columns: policy details, insured person, incident details, claim amounts, vehicle and the label `fraud_reported` (Y/N)
* 24.7% of claims are labelled fraudulent
* Missing values are coded as `?` in the CSV; they are replaced by the most common value for `collision_type`, `property_damage` and `police_report_available`
* Source: a widely used public insurance claims dataset; original source *to be added*

---

## Exploratory Analysis

**Total claim amount**
![1](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/f413f61b-fa83-4736-bbce-6d980c5f0cfd)
![3](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/12e5c9eb-870a-45d3-9a48-240688a9c0dc)

**Incident types**
![2](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/612e49d2-000c-4d55-b98c-8331db3c510c)

Multi-vehicle and single-vehicle collisions are the most common incident types.

**Age vs total claim amount**
![4](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/6ff26668-e5d2-49f9-8a6a-35fff3ee7026)

No clear relationship between age and claim amount.

---

## What Differs in Fraudulent Claims

**Numeric features (mean values)**

| Feature | Not fraud | Fraud |
| --- | --- | --- |
| Total claim amount | 50,289 | 60,302 |
| Vehicle claim | 36,090 | 43,534 |
| Property claim | 7,019 | 8,560 |
| Injury claim | 7,179 | 8,208 |

Claim amounts have the strongest correlation with fraud, but it is weak (r = 0.17 for vehicle claim, 0.16 for total claim). Other numeric features differ very little.

**Categorical features (share of claims that are fraudulent)**

| Feature | Category | Fraud rate |
| --- | --- | --- |
| Incident severity | Major Damage | **61%** |
| | Total Loss | 13% |
| | Minor Damage | 11% |
| | Trivial Damage | 7% |
| Insured hobby | chess | **83%** |
| | cross-fit | **74%** |
| | all other hobbies | 9% to 30% |
| Authorities contacted | Other / Ambulance / Fire / Police | 32% / 29% / 27% / 21% |

Incident severity and two hobbies stand out strongly. These are patterns in this dataset and should not be read as causes of fraud.

---

## Fraud Prediction Model

* Random Forest classifier (scikit-learn defaults, `random_state=42`) on all columns, with categorical columns label-encoded
* 70/30 train and test split

| Metric (test set, 300 claims) | Value |
| --- | --- |
| Accuracy | 0.76 |
| ROC-AUC | 0.83 |
| Fraud precision | 0.59 |
| **Fraud recall** | **0.30** |

The model finds only 30% of fraudulent claims. Accuracy looks reasonable mainly because 73% of test claims are not fraud.

**Feature importance**
![5](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/4b41fdf4-9d1f-46f5-bc26-0ac9e22add9c)

Incident severity is by far the most important feature (0.19), followed by insured hobbies (0.07). Several identifiers such as `policy_number`, `insured_zip` and `incident_location` also receive some importance, which suggests the model is partly fitting noise.

The notebook also draws box plots of the top features by fraud label. For categorical features (incident severity, hobbies, authorities contacted, zip code) these plots show label-encoded integer codes, so they are hard to interpret; the fraud-rate table above is the clearer summary.

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Insurance_Claims.ipynb
```

Run all cells in order.

---

## Limitations and Next Steps

* Remove identifier columns (policy number, zip code, location, raw dates) before modelling
* Use one-hot encoding for categorical features
* Handle class imbalance (class weights or threshold tuning) to raise fraud recall
* Use cross-validation instead of a single split

---

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn
