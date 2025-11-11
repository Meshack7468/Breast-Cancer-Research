# BREAST CANCER RESEARCH
***
## OVERVIEW
This project applies machine learning to predict breast cancer molecular subtypes and patient survival outcomes. It provides a cost-effective, data-driven alternative to expensive molecular tests like PAM50, helping clinicians make informed treatment and prognosis decisions.

## PROBLEM STATEMENT
This project aims to predict breast cancer subtypes and survival outcomes using machine learning, providing an accessible alternative for hospitals without advanced genetic tests like PAM50 to support better treatment decisions.

## Business Objectives
This project aims to create machine learning models that deliver clinical value by:

* Molecular Subtype Prediction: Developing a model to predict PAM50 + Claudin-low breast cancer subtypes, providing a cost-effective digital alternative to genetic testing for guiding personalized treatment.
* Binary Survival Prediction: Predicting patient survival status (Living or Deceased) to help identify high-risk patients early for closer monitoring or targeted therapy.
* Multi-class Vital Status Prediction: Classifying survival outcomes as Died of Disease, Died of Other Causes, or Living to distinguish cancer-related deaths and support deeper clinical insight.

## Data Understanding
The dataset comes from the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) study — a well-known clinical and genomic breast cancer dataset containing information on 2,509 patients. Each record represents a patient, with 39 features (12 numerical and 27 categorical). The data is stored in a TSV (tab-separated values) format.

Key Features:
* Pam50 + Claudin-low subtype: Molecular subtype classification (Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like).
* Nottingham Prognostic Index (NPI): Combines tumor size, grade, and lymph node status for prognosis prediction.
* Age at Diagnosis, Tumor Size, and Tumor Stage: Describe patient demographics and tumor characteristics.
* ER, PR, HER2 Status: Biomarkers critical for determining subtype and treatment response.
* Therapy Indicators: Whether the patient received chemotherapy, hormone therapy, or radiotherapy.
* Lymph Nodes Examined Positive: Shows whether the cancer has spread to the lymph nodes.
* Neoplasm Histologic Grade & Cellularity: Checks tumor aggressiveness.
* Overall Survival & Relapse-Free Status: Capture survival duration and recurrence of the disease.

Data Quality:
Some columns contain missing values (e.g., Tumor Stage – 721, Type of Surgery – 554, Cellularity – 592).
Missing values in numerical columns are imputed with the median, while categorical columns with missing values are dropped.

Target Variables:
* Pam50 + Claudin-low subtype → Multi-class target for molecular subtype prediction.
* Overall Survival Status → Binary target for survival prediction (Living / Deceased).
* Patient’s Vital Status → Multi-class target for classifying survival outcomes (Living / Died of Disease / Died of Other Causes).


## METHODS
This project uses predictive statistics and descriptive statistics to analyze data, reveal key clinical insights, and build machine learning models that predict breast cancer subtypes and patient survival outcomes.

## EDA
#### Molecular Subtype vs Survival Status

![alternative](./Images/SubtypeVsSurvival.png)

* Comparing molecular subtypes vs survival we observe that Luminal A is associated with best survival status (highest number of living patients) this may suggest that it respond well to treatment

#### Comparing How Different Molecular Subtype Respond to Chemotherapy

![alternative](./Images/SubtypevsChemotherapy.png)

* Luminal A show the highest resistance to chemotherapy, indicating a low response rate to this type of treatment followed by Luminal B.
* Basal and HER2 subtypes show good responsiveness to chemotherapy.

#### Comparing How Different Molecular Subtype Respond to Hormone therapy

![alternative](./Images/SubtypevsHormoneTherapy.png)

* Luminal A and Luminal B show the best response to hormone therapy. These Pam50 subtypes depend on estrogen or progesterone hormone to grow, and they are often treated by blocking these hormones

#### ER and PR Status VS Survival rate

![alternative](./Images/ERPRVsSurvival.png)

* ER+,PR+ have higher survival rates compared to ER-,PR-. ER+ and PR+ have the highest number of living patients and the gap between living patients bar and
lower score for patients who died of disease. ER-,PR- show poor prognosis

#### How Tumor Stage Influnce Survival Status

![alternative](./Images/TumorStageVsSurvival.png)

Lower stages (0 and 1) have better survival rates compared to higher stages(2 and 3)

#### How Age at Diagnosis Influence Survival

Most Deceased patients are of older age. From the plot there are some outliers for the DECEASED class suggesting that there is lower survival rates for some younger women and older women. Middle-aged women (45-70) show the best survival rates. 

## RESULTS
### Molecular subtype prediction
Classify tumors into molecular subtypes using clinical and genetic features 
We build 2 models for this task, Random Forest and XGBoost
* Models performance comparisons.
Random Forest acheived an Accuracy of 67% and Macro F1-score of 56%
XGBoost achieved an Accuracy of 67% and Macro F1-score of 54%
Random Forest is the Best performing model because it has highest Macro F1-score. We compare using Macro F1-score because the classes are imbalanced and we desire to treat each classes equally

### Survival Status Prediction
Build models that Predict whether a patient is alive or deceased this will allow us to identify high-risk patients who may need closer monitoring or aggressive therapy.
* Models performance comparisons
The Random Forest model performed best for this prediction compared to Logistic Regression model. It achieved an Accuracy of 73% and 
ROC-AUC of 79% while Logistic Regression model have an Accuracy of 68% and ROC-AUC of 77%, we compare using ROC-AUC because this is a binary classification
Both models are better at predicting deceased patients than living patients, which is clinically important for identifying high-risk individuals.
Random Forest’s higher ROC-AUC shows it can better capture complex interactions between clinical and molecular features.

### Vital Status Prediction
This will help distinguish cancer-related deaths and death from other causes.
It also reveals which clinical or molecular features are most associated with cancer-specific mortality.
It also help us identify high-risk patients and knowing who is at risk of dying from cancer allows closer monitoring and guided treatment approach.
For this part we build a Random Forest, XGBoost and Tuned XGBoost model
* Models performance comparisons
The XGBoost performed better with an Accuracy of 58%, macro F1 score of 0.57 it performed better on the minority “DIED OF OTHER CAUSES” class.


## CONCLUSIONS
1. Integrate these tools into hospital workflows
* Use survival and vital prediction model to identify high-risk patients and guide individualized monitoring and treatment decisions.
* Use molecular subtype prediction model to estimate tumor aggressiveness when full genetic testing is unavailable.
2. Use feature importance analysis insight to prioritize High-Risk Patients:
* Older patients(Above 75) , those with larger tumors, higher grade, more positive lymph nodes, should receive closer monitoring.
3. Personalize treatment planning for each subtype:
* Luminal A Respond best hormone therapy
* Luminal B: Respond best to combined hormone and chemo therapy
* HER2-enriched: Require targeted therapy (e.g., trastuzumab, pertuzumab) these significantly improve survival beyond what surgery alone can achieve.
* Basal-like: Aggressive tumors that need intensive targeted systemic therapy surgery alone isn’t enough for effective treatment
* Claudin-low: Benefit most from biological treatments they show limited response to surgery or chemotherapy alone.

****

## Repository Structure
```
├── 
├── 
├── README.md
├── 
└── presentation.pdf
```
