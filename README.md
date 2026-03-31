# 🌐 Visa Approval Classification using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Predicting U.S. Visa Certification Outcomes using Ensemble Machine Learning Models — built for EasyVisa & the Office of Foreign Labor Certification (OFLC).**

---

## 📌 Project Overview

The U.S. visa application process has grown increasingly complex, with the Office of Foreign Labor Certification (OFLC) processing nearly **776,000 employer applications** in FY 2016 alone — a 9% year-over-year increase. Manually reviewing every case is no longer scalable.

This project develops a **binary classification machine learning solution** to predict whether a visa application will be **Certified** or **Denied**, enabling data-driven shortlisting of high-potential candidates and reducing the review burden on OFLC officers.

👉 [Open the notebook to explore full analysis](notebook/Visa_Approval_Classification_using_Machine_Learning.ipynb)

---

## 💼 Business Problem

### Real-World Context

Under the **Immigration and Nationality Act (INA)**, U.S. employers can hire foreign workers for roles that cannot be filled by the domestic workforce. The OFLC certifies such applications only when employers demonstrate a genuine labor shortage and offer competitive, prevailing wages.

As applications increase year-on-year, the manual review pipeline becomes a critical bottleneck — delaying decisions, increasing costs, and introducing inconsistency in outcomes.

### Stakeholders

- **OFLC** — Needs an automated pre-screening layer to prioritize and triage incoming cases efficiently.
- **EasyVisa** — A data science consultancy hired to build the predictive solution and extract actionable workforce insights.
- **Employers & Applicants** — Benefit from faster, more transparent, and consistent certification decisions.

### Decision Impact

A well-performing model can flag high-probability approvals early, enabling OFLC to fast-track deserving applications, identify denial patterns, and improve labor market outcomes for employers and foreign workers alike.

---

## 📂 Dataset

| Attribute | Details |
|---|---|
| **Source** | Office of Foreign Labor Certification (OFLC) — EasyVisa ML Project |
| **Size** | 25,480 rows × 12 columns |
| **Target Variable** | `case_status` (Certified = 1, Denied = 0) |
| **Class Distribution** | Certified: 17,018 (66.8%) / Denied: 8,462 (33.2%) |

### Key Features

| Feature | Type | Description |
|---|---|---|
| `continent` | Categorical | Continent of the employee (6 categories) |
| `education_of_employee` | Categorical | Education level: High School, Bachelor's, Master's, Doctorate |
| `has_job_experience` | Binary | Prior job experience (Y/N) |
| `requires_job_training` | Binary | Whether job training is required (Y/N) |
| `no_of_employees` | Numerical | Number of employees in the employer's company |
| `yr_of_estab` | Numerical | Year of establishment of the employer's company |
| `region_of_employment` | Categorical | U.S. region of intended employment (5 regions) |
| `prevailing_wage` | Numerical | Average wage for the occupation in the area |
| `unit_of_wage` | Categorical | Wage frequency: Hour, Week, Month, Year |
| `full_time_position` | Binary | Full-time (Y) or part-time (N) |

---

## 🔬 Methodology

### 1. Data Understanding & Cleaning

The analysis began with a thorough examination of the dataset structure. The dataset contains **25,480 records and 12 columns**, comprising 3 numerical and 9 categorical variables. Upon inspection, the data was found to be remarkably clean — **zero missing values, zero duplicates, and zero null entries** across all columns. The `case_id` column was identified as a non-informative unique identifier and subsequently dropped before modeling.

A statistical summary revealed important distribution characteristics. The average employer had approximately **5,667 employees**, but with a high standard deviation of 22,877, reflecting the presence of both small businesses and large corporations. The `prevailing_wage` ranged from $2.13 to over $319,210, with a mean of $74,456, indicating wage diversity. The `yr_of_estab` spanned from 1800 to 2016, with a median of 1997, confirming that most companies in the dataset are relatively modern.

### 2. Exploratory Data Analysis

**Univariate Analysis** was conducted across all numerical and categorical variables using histograms, boxplots, and labeled bar charts.

For numerical variables, all three — `no_of_employees`, `yr_of_estab`, and `prevailing_wage` — exhibited **right-skewed distributions** with notable high-value outliers. The employee count distribution was particularly extreme, with most companies clustered near zero while a few outliers exceeded 500,000 employees. Prevailing wage showed a bimodal-like shape with a large spike at very low values (attributable to hourly-wage roles) and a gradual tail into high-salary positions.

For categorical variables, **Asia** dominated the continent distribution with 16,861 cases. **Bachelor's degree** holders were the most common education level (10,234), followed closely by Master's (9,634). The majority of applicants — 14,802 — had prior job experience, 22,525 did not require additional job training, and 22,773 applied for full-time positions. Among the two case outcomes, **Certified** was the majority class with 17,018 cases.

**Bivariate Analysis** was conducted via correlation heatmaps for numerical variables and stacked bar charts for categorical variable interactions with the target.

The correlation heatmap among numerical variables revealed **near-zero correlations** between all three — `no_of_employees` vs `yr_of_estab` (−0.02), `prevailing_wage` vs `yr_of_estab` (0.01) — confirming no multicollinearity concerns.

Bivariate plots against `case_status` surfaced several important patterns: employees with **higher educational qualifications** (Master's and Doctorate) had notably higher certification rates, while High School graduates had a majority of their applications denied. Applicants **with job experience** were certified at a significantly higher rate (74.5%) compared to those without (56.2%). The **Midwest region** showed the highest certification rate among all employment regions. **Yearly wage** units were strongly associated with certification, while hourly-wage applications had a disproportionately higher denial rate. Both full-time and part-time positions showed similar certification proportions, with full-time having a slight edge.

### 3. Data Preprocessing

An outlier check was performed on all three numerical columns. Upon examination, the observed outliers in `no_of_employees`, `yr_of_estab`, and `prevailing_wage` were determined to be **domain-realistic** (very old companies, very large firms, and hourly-vs-annual wage differences respectively) and were **retained without treatment**.

The target variable `case_status` was **label-encoded** (Denied → 0, Certified → 1). The dataset was split into training and validation sets using `train_test_split` (80/20 split). All categorical variables were then transformed using **one-hot encoding** via `pd.get_dummies`, expanding the feature space from 10 to 21 columns.

To address the **class imbalance** (approximately 67:33 split), two resampling strategies were evaluated alongside the original data: **SMOTE oversampling** (using `imblearn.over_sampling.SMOTE` with `k_neighbors=5`) to balance the minority class synthetically, and **Random Under Sampling** (using `imblearn.under_sampling.RandomUnderSampler`) to reduce the majority class.

### 4. Model Building

Five ensemble classification algorithms were trained and evaluated on all three data variants — original, oversampled, and undersampled:

- **Bagging Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**

The **primary evaluation metric was Recall**, given the business context: a False Negative (predicting Denied when the case should be Certified) represents a missed opportunity for a deserving applicant and carries significant operational and legal consequences for employers.

On the original data, **Random Forest achieved a perfect training recall of 1.0 and a validation recall of 0.968**, outperforming all other models. XGBoost was a strong second with a validation recall of 0.919. On oversampled data, results were similar with Random Forest leading at 0.962 validation recall. On undersampled data, all models showed weaker generalization, though Random Forest continued to lead at 0.814 validation recall.

### 5. Hyperparameter Tuning & Model Selection

Three models were selected for hyperparameter optimization using `RandomizedSearchCV` with recall as the scoring metric:

**Tuned AdaBoost (Original Data)** — Best parameters: `n_estimators=50`, `learning_rate=0.01`, `base_estimator=DecisionTreeClassifier(max_depth=3)`. Achieved a training recall of 0.914 and validation recall of 0.916, with an F1 of 0.824 — a well-balanced performer.

**Tuned Gradient Boosting (Undersampled Data)** — Best parameters: `subsample=0.7`, `n_estimators=75`, `max_features=0.5`, `learning_rate=0.05`, `init=AdaBoostClassifier`. Achieved a training recall of 0.767 and validation recall of 0.764, with strong precision of 0.818 and an F1 of 0.790 — demonstrating good generalization on held-out data.

**Tuned Gradient Boosting (Original Data)** — Best parameters: `subsample=0.9`, `n_estimators=100`, `max_features=1`, `learning_rate=0.01`, `init=AdaBoostClassifier`. Achieved near-perfect recall (0.995) on both training and validation, but with lower precision (0.677), indicating a high volume of false positives.

**Final Model Selected: Gradient Boosting trained on Undersampled Data**, as it offers the best overall balance of precision (0.818), recall (0.764), and F1 (0.790) — making it the most reliable choice for real-world deployment where both false positives and false negatives carry meaningful costs. On the unseen test set, this model achieved an accuracy of 0.724, recall of 0.758, precision of 0.816, and F1 of 0.786.

### 6. Feature Importance

Analysis of feature importances from the final Gradient Boosting model revealed the following hierarchy of influential predictors:

- `education_of_employee_High School` — highest relative importance (~0.25), underscoring how lacking higher education significantly impacts denial risk
- `has_job_experience_Y` — second most important (~0.21), confirming that prior work experience is a primary positive signal
- `education_of_employee_Master's` — third, reflecting the premium placed on advanced degrees
- `prevailing_wage` — fourth, showing that competitive wage offers correlate with higher approval likelihood
- `education_of_employee_Doctorate`, `continent_Europe`, `unit_of_wage_Year`, and `region_of_employment_Midwest` — contributing moderate secondary influence

---

## 📊 Key Results

| Model | Dataset | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Tuned AdaBoost | Original | 0.739 | **0.916** | 0.749 | 0.824 |
| Tuned GBM | Undersampled | 0.728 | 0.764 | **0.818** | **0.790** |
| Tuned GBM | Original | 0.679 | **0.995** | 0.677 | 0.806 |
| **Final Model (GBM Undersampled) on Test Set** | Unseen | **0.724** | **0.758** | **0.816** | **0.786** |

**Key Insights:**
- Education level (particularly High School vs. advanced degree) is the single most important predictor of visa denial
- Prior job experience nearly doubles the odds of certification
- Applicants paid on an annual basis have significantly higher certification rates than hourly-wage applicants
- Midwest and West regions consistently show higher approval rates than Island or Northeast regions
- The dataset exhibits class imbalance (~67% Certified), requiring careful metric selection and resampling strategies

---

## 💡 Business Impact

**1. Automated Pre-Screening at Scale:** Deploying the Gradient Boosting model as a pre-screening layer can significantly reduce the volume of cases requiring manual review, allowing OFLC officers to focus on genuinely borderline applications rather than processing the entire queue.

**2. Education-Driven Recruitment Strategy:** Companies should prioritize sourcing candidates with Master's or Doctorate qualifications for foreign labor certification applications. EasyVisa can use education-level filters during initial candidate intake to improve certification yield before applications are even filed.

**3. Experience-First Hiring Criteria:** Since prior job experience is the second most decisive factor, employers should systematically document and highlight verifiable work history in applications. OFLC guidance could formalize minimum experience thresholds for faster-track processing.

**4. Wage Competitiveness as a Strategic Lever:** Certified cases trend toward higher prevailing wages and annual wage structures. Employers offering competitive, clearly annual-basis salaries should communicate this explicitly, as it positively signals intent to comply with prevailing wage protections.

**5. Regional Workforce Planning:** Organizations operating in the Midwest or West benefit from structurally higher approval rates. Companies with geographic flexibility in role placement should consider routing foreign labor requests toward these regions when operationally feasible, in collaboration with OFLC's regional offices.

---

## 🛠️ Skills

### Technical Skills

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AA4A44?style=for-the-badge&logo=xgboost&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AC1D?style=for-the-badge&logo=seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Exploratory Data Analysis](https://img.shields.io/badge/Data_Analysis-FFA500?style=for-the-badge&logo=google-analytics&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

### Soft Skills

![Analytical Thinking](https://img.shields.io/badge/Analytical_Thinking-4B0082?style=for-the-badge&logo=mindmap&logoColor=white)
![Communication](https://img.shields.io/badge/Communication-25D366?style=for-the-badge&logo=google-messages&logoColor=white)
![Problem Solving](https://img.shields.io/badge/Problem_Solving-FF4500?style=for-the-badge&logo=brainly&logoColor=white)
![Attention to Detail](https://img.shields.io/badge/Attention_to_Detail-00CED1?style=for-the-badge&logo=google-search-console&logoColor=white)

---

## 🧠 Key Learnings

- **Class Imbalance is not just a modeling problem — it is a business problem.** Understanding what a false negative means in the real world (a wrongly denied applicant) fundamentally shapes metric choices and resampling strategy decisions.
- **Ensemble methods offer complementary strengths.** Random Forest generalizes well with raw data but may overfit; Gradient Boosting with undersampling provides better balance between precision and recall in production settings.
- **Feature importance is more than a ranking — it's a story.** The dominance of education and job experience in the model mirrors OFLC's policy priorities, validating that the model has learned something genuinely meaningful rather than spurious patterns.
- **Hyperparameter tuning via `RandomizedSearchCV` with a domain-appropriate scorer** (recall, not accuracy) is critical. Optimizing for accuracy on an imbalanced dataset would have produced a misleadingly high-scoring but operationally useless model.
- **Bivariate EDA before modeling saves significant effort downstream.** Key signals like the wage unit vs. case_status relationship would have been missed with purely univariate analysis, yet they emerged as important model features.

---

## 🚀 Future Improvements

1. **Calibrated Probability Scoring:** Move beyond binary classification to output calibrated probability scores for each application, enabling OFLC to set dynamic thresholds for prioritization rather than hard pass/fail decisions.

2. **SHAP-based Explainability Layer:** Integrate SHAP (SHapley Additive exPlanations) to provide per-application explainability, making the model auditable and defensible for regulatory compliance and applicant communication.

3. **Temporal Drift Monitoring:** Incorporate time-series validation to detect model drift as labor market conditions evolve — the features important for certification in 2016 may shift with immigration policy changes over time.

4. **Natural Language Processing on Job Descriptions:** If job description text data becomes available, applying NLP embeddings could add a rich new signal dimension that the current structured dataset lacks.

5. **Deployment as an API Microservice:** Wrap the final model in a FastAPI or Flask microservice to allow EasyVisa's application intake systems to call real-time predictions during the application submission phase, enabling on-the-fly feedback to employers.

---

## 👨‍💻 Author

**Nabankur Ray**

Passionate about real-world data-driven solutions

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/nabankur14) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nabankur-ray-876582181/)

![GitHub Stats](https://github-readme-stats-eight-theta.vercel.app/api?username=nabankur14&show_icons=true)

---

⭐ If you like this project — Give it a ⭐ on GitHub — it helps a lot!