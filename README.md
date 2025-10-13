<h1 align="center" style="color:#2b7a78;">Visa Case Outcome Prediction – EasyVisa Analytics</h1>
<h3 align="center" style="color:#17252a;">Predicting US Visa Certification Decisions Using Machine Learning to Enhance Efficiency and Policy Insights</h3>

<p align="center">
  <strong>Author:</strong> <a href="https://github.com/nabankur14" target="_blank" style="color:#3aafa9;">Nabankur Ray</a>  
</p>

<hr>

<h2 style="color:#17252a;">Overview</h2>
<p>
This project applies <strong>Machine Learning</strong> and <strong>Data Analytics</strong> to predict visa case outcomes 
(<em>Certified</em> or <em>Denied</em>) for the <strong>EasyVisa</strong> initiative under the Office of Foreign Labor Certification (OFLC). 
The objective is to assist officials in automating applicant shortlisting, reducing manual workload, and identifying 
factors that influence visa approvals. Comprehensive <em>Exploratory Data Analysis (EDA)</em>, <em>sampling strategies</em>, 
and <em>ensemble modeling</em> were employed to build accurate, interpretable, and business-relevant predictions.
</p>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Objective</summary>
  <p>
  The primary goals of this project are to:
  <ul>
    <li>Analyze applicant and employer factors influencing visa certification outcomes.</li>
    <li>Develop predictive ML models to classify applications as <strong>Certified</strong> or <strong>Denied</strong>.</li>
    <li>Address class imbalance using <strong>over-sampling</strong> and <strong>under-sampling</strong> techniques.</li>
    <li>Compare ensemble algorithms (Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost) to select the best performer.</li>
    <li>Provide business insights and recommendations to optimize policy and decision-making.</li>
  </ul>
  </p>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Dataset</summary>
  <ul>
    <li><strong>Source:</strong> EasyVisa / OFLC visa application dataset (U.S. Employment Visa Cases).</li>
    <li><strong>Size:</strong> 25,480 rows × multiple features.</li>
    <li><strong>Features:</strong>
      <ul>
        <li><code>continent</code> – Applicant’s continent of origin.</li>
        <li><code>education_of_employee</code> – Highest education level attained.</li>
        <li><code>has_job_experience</code> – Whether the applicant has relevant work experience.</li>
        <li><code>requires_job_training</code> – Indicates if training is required for the job.</li>
        <li><code>no_of_employees</code> – Total employees at employer’s organization.</li>
        <li><code>region_of_employment</code> – U.S. region where the applicant would be employed.</li>
        <li><code>prevailing_wage</code> – Offered salary for the position.</li>
        <li><code>full_time_position</code> – Employment type (Full-time/Part-time).</li>
        <li><code>case_status</code> – Target variable (Certified / Denied).</li>
      </ul>
    </li>
    <li><strong>Data Quality:</strong> Clean dataset with no missing values; categorical and numerical features validated.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Methodology</summary>
  <ol>
    <li><strong>Data Understanding & Cleaning:</strong> Verified missing values, outliers, and encoded categorical variables.</li>
    <li><strong>Exploratory Data Analysis (EDA):</strong> Conducted univariate and bivariate visualizations using barplots, boxplots, and heatmaps to identify influential features.</li>
    <li><strong>Sampling Techniques:</strong> Applied both <strong>oversampling</strong> and <strong>undersampling</strong> to address class imbalance.</li>
    <li><strong>Model Building & Evaluation:</strong> Implemented ensemble algorithms:
      <ul>
        <li>Bagging Classifier</li>
        <li>Random Forest Classifier</li>
        <li>AdaBoost Classifier</li>
        <li>Gradient Boosting Classifier</li>
        <li>XGBoost Classifier</li>
      </ul>
    </li>
    <li><strong>Hyperparameter Tuning:</strong> Optimized ensemble parameters and compared models on accuracy, recall, precision, and F1-score.</li>
    <li><strong>Feature Importance:</strong> Identified top predictors such as <code>education_of_employee</code>, <code>has_job_experience</code>, and <code>prevailing_wage</code>.</li>
  </ol>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Tools & Technologies</summary>
  <p>
  <code>Python</code>, <code>Pandas</code>, <code>NumPy</code>, <code>Scikit-learn</code>, <code>XGBoost</code>,  
  <code>Matplotlib</code>, <code>Seaborn</code>, <code>Jupyter Notebook</code>
  </p>
</details>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Results & Insights</summary>
  <ul>
    <li><strong>Certification Rate:</strong> ~67% of cases were certified; 33% were denied.</li>
    <li><strong>Top Predictors:</strong> <code>education_of_employee</code>, <code>has_job_experience</code>, and <code>prevailing_wage</code>.</li>
    <li><strong>Best Model:</strong> Tuned <strong>Gradient Boosting</strong> on undersampled data — best balance of recall and precision.</li>
    <li><strong>Insights:</strong>
      <ul>
        <li>Higher education and experience increase certification likelihood.</li>
        <li>Applicants from certain regions show higher denial trends due to skill mismatch or wage variance.</li>
        <li>Prevailing wage and job training needs significantly influence decisions.</li>
      </ul>
    </li>
    <li><strong>Business Recommendations:</strong>
      <ul>
        <li>Prioritize applicants with advanced education and relevant experience.</li>
        <li>Standardize wage offers to improve certification consistency.</li>
        <li>Enhance employer training policies to support applicants from underrepresented regions.</li>
      </ul>
    </li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Future Scope</summary>
  <ul>
    <li>Deploy the model as an <strong>interactive dashboard</strong> or <strong>API</strong> for automated screening.</li>
    <li>Integrate additional socio-economic or employer-level data for deeper insights.</li>
    <li>Explore advanced ensembles (Stacking, LightGBM, CatBoost) to enhance performance.</li>
    <li>Develop a <strong>Tableau</strong> or <strong>Power BI</strong> visualization for visa trends and model explanations.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Key Learnings</summary>
  <ul>
    <li>Built a complete ML pipeline from data exploration to business recommendations.</li>
    <li>Gained practical understanding of <strong>class imbalance handling</strong> and <strong>ensemble model tuning</strong>.</li>
    <li>Enhanced skills in <strong>model interpretation</strong> and translating insights into <strong>policy strategies</strong>.</li>
    <li>Strengthened expertise in <strong>EDA, evaluation metrics,</strong> and <strong>data-driven storytelling</strong>.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Folder Structure</summary>
  <pre style="background:#f0f0f0; padding:10px; border-radius:8px;">

visa_case_prediction_ml_project/
│
├── EasyVisa.csv                                      → Raw and processed visa case datasets  
├── Visa_Approval_Classification_using_Machine_Learning.ipynb           → Main Jupyter Notebook (EDA + ML modeling)
├── Employment Visa Analytics_Predicting_Certification_Outcomes.pdf     → Full business & analytical report
└──  README.md                                   → Project documentation (this file)            
  </pre>
</details>

<p align="center" style="color:#555;">
>>> All project files are organized and accessible for easy reproducibility and analysis.
</p>

<h2 style="color:#17252a;"> #Tags</h2>
<p>
#MachineLearning #DataScience #VisaAnalytics #Classification #EnsembleLearning #XGBoost #Python #FeatureImportance #BusinessIntelligence #PredictiveModeling #PolicyOptimization
</p>

<hr>
<p align="center" style="font-size:14px; color:#555;">
© 2025 <strong>Nabankur Ray</strong> | Data Scientist
</p>
