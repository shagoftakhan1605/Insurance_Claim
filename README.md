# Insurance Claims Data Analysis and Fraud Detection

This repository contains a Jupyter notebook that performs an analysis of insurance claims data to identify trends, detect anomalies, and identify common factors in fraudulent claims.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Notebook Overview](#notebook-overview)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
Insurance fraud is a significant issue that impacts insurance companies and policyholders alike. By analyzing insurance claims data, we can identify patterns and anomalies that may indicate fraudulent activity. This analysis helps in improving claims processing efficiency and detecting fraud early.

## Dataset
The dataset used in this analysis is a hypothetical insurance claims dataset. It includes information on policy details, insured individuals, incident details, claim amounts, and whether the claim was reported as fraudulent.

## Dependencies
The following Python libraries are required to run the notebook:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Notebook Overview
The notebook is divided into several steps:

1. **Load the Dataset:** Load the insurance claims dataset and display the first few rows to understand its structure.

2. **Clean and Preprocess the Dataset:** Handle missing values and preprocess the dataset for analysis.

3. **Exploratory Data Analysis (EDA):** Analyze trends in claim types and amounts, and visualize the distributions.

4. **Anomaly Detection:** Detect anomalies in the total claim amounts and other features using box plots and scatter plots.

5. **Analyze Claims Processing Times:** (If available) Analyze and visualize claims processing times to identify workflow bottlenecks.

6. **Analysis of Fraudulent Claims:** Identify common factors in fraudulent claims using correlation analysis and feature importance from a Random Forest classifier.

7. **Visualization of Key Factors:** Visualize the most important features that predict fraud and their distributions for fraudulent and non-fraudulent claims.

## Results

### Step 1: Load the Dataset
The dataset is loaded successfully, and the first few rows are displayed.

### Step 2: Clean and Preprocess the Dataset
Missing values are handled, and the dataset is preprocessed for analysis.

### Step 3: Exploratory Data Analysis (EDA)

#### 1. Distribution of Total Claim Amounts
![1](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/f413f61b-fa83-4736-bbce-6d980c5f0cfd)
- This histogram shows the distribution of total claim amounts. The KDE plot indicates that there is a significant number of low-value claims and a few high-value claims.

#### 2. Count of Different Incident Types
![2](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/612e49d2-000c-4d55-b98c-8331db3c510c)
- This bar chart shows the frequency of different types of incidents. Multi-vehicle collisions and single-vehicle collisions are the most common types of incidents reported.

### Step 4: Anomaly Detection

#### 3. Box Plot of Total Claim Amounts
![3](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/12e5c9eb-870a-45d3-9a48-240688a9c0dc)
- The box plot visualizes the spread of total claim amounts and highlights any outliers. Most claims fall within a certain range, but there are some high-value outliers.

#### 4. Scatter Plot of Age vs. Total Claim Amount
![4](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/6ff26668-e5d2-49f9-8a6a-35fff3ee7026)
- This scatter plot shows the relationship between the age of the insured and the total claim amount. There is no clear trend, but some high-value claims are scattered across different ages.

### Step 5: Analyze Claims Processing Times
- The dataset does not contain a `claims_processing_time` column.

### Step 6: Analysis of Fraudulent Claims

#### Comparison of Fraudulent and Non-Fraudulent Claims
- This table shows the average values of various features for fraudulent and non-fraudulent claims. Fraudulent claims tend to have higher policy deductibles, umbrella limits, total claim amounts, injury claims, property claims, and vehicle claims.

#### Detailed Analysis:
```plaintext
            months_as_customer        age  policy_number  policy_deductable  \
fraud_reported                                                             
N                   202.600266  38.884462  550571.297477      1130.810093   
Y                   208.080972  39.141700  533030.206478      1151.821862   

            policy_annual_premium  umbrella_limit    insured_zip  \
fraud_reported                                                     
N                     1258.430000    1.023904e+06  500419.537849   
Y                     1250.236275    1.336032e+06  503637.959514   

            capital-gains  capital-loss  incident_hour_of_the_day  \
fraud_reported                                                     
N             25432.005312 -26554.581673                 11.626826   
Y             24193.522267 -27522.672065                 11.696356   

            number_of_vehicles_involved  bodily_injuries  witnesses  \
fraud_reported                                                    
N                              1.808765        0.976096   1.455511   
Y                              1.931174        1.040486   1.582996   

            total_claim_amount  injury_claim  property_claim  vehicle_claim  \
fraud_reported                                                                
N                50288.605578   7179.229748    7018.884462   36090.491368   
Y                60302.105263   8208.340081    8560.121457   43533.643725   

            auto_year  
fraud_reported      
N          2005.075697  
Y          2005.186235  
```

#### Correlation Analysis
- The correlation analysis identifies which features are most strongly correlated with the binary fraud indicator. Vehicle claim amount, total claim amount, and property claim amount are among the top correlated features.

#### Detailed Correlations:
```plaintext
fraud_reported_binary          1.000000
vehicle_claim                  0.170049
total_claim_amount             0.163651
property_claim                 0.137835
injury_claim                   0.090975
umbrella_limit                 0.058622
number_of_vehicles_involved    0.051839
witnesses                      0.049497
bodily_injuries                0.033877
months_as_customer             0.020544
insured_zip                    0.019368
policy_deductable              0.014817
age                            0.012143
auto_year                      0.007928
incident_hour_of_the_day       0.004316
policy_annual_premium         -0.014480
capital-loss                  -0.014863
capital-gains                 -0.019173
policy_number                 -0.029443
Name: fraud_reported_binary, dtype: float64
```

#### Feature Importance Using Random Forest
![5](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/4b41fdf4-9d1f-46f5-bc26-0ac9e22add9c)
- The Random Forest classifier identifies the most important features for predicting fraud. Incident severity, insured hobbies, and property claim amount are among the top features.

#### Detailed Feature Importances:
```plaintext
incident_severity              0.185104
insured_hobbies                0.066129
property_claim                 0.041000
policy_annual_premium          0.039106
insured_zip                    0.036975
policy_number                  0.036216
vehicle_claim                  0.035269
incident_date                  0.034826
injury_claim                   0.033508
months_as_customer             0.032641
total_claim_amount             0.032026
incident_location              0.030942
policy_bind_date               0.028961
auto_model                     0.028269
insured_occupation             0.028186
age                            0.027887
incident_hour_of_the_day       0.025587
auto_year                      0.025457
auto_make                      0.020890
umbrella_limit                 0.020252
capital-gains                  0.019259
capital-loss                   0.018800
incident_state                 0.015671
insured_relationship           0.015431
incident_city                  0.015202
insured_education_level        0.014838
authorities_contacted          0.013155
witnesses                      0.008814
policy_csl                     0.008707
bodily_injuries                0.008595
policy_state                   0.007912
collision_type                 0.007856
policy_deductable              0.007526
number_of_vehicles_involved    0.007223
incident_type                  0.006905
insured_sex                    0.005748
property_damage                0.004975
police_report_available        0.004152
dtype: float64
```

### Step 7: Visualization of Key Factors











#### 8. Distribution of Incident Severity for Fraudulent and Non-Fraudulent Claims
![6](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/ce89e3fd-ffee-4475-abfc-aee6a2fe9707)
- This box plot compares the distribution of incident severity for fraudulent and non-fraudulent claims. Fraudulent claims tend to have higher incident severity.

#### 9. Distribution of Insured Hobbies for Fraudulent and Non-Fraudulent Claims
![7](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/9cb636fb-8626-47a0-be0a-51d0f0139b8e)
- This box plot shows the distribution of insured hobbies for fraudulent and non-fraudulent claims. There is a slight difference in the hobbies of insured individuals who report fraudulent claims.

#### 10. Distribution of Property Claim for Fraudulent and Non-Fraudulent Claims
![8](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/d589cc5f-aff6-46a0-b304-e6aecee77e97)
- This box plot compares the distribution of property claim amounts for fraudulent and non-fraudulent claims. Fraudulent claims tend to have higher property claim amounts.

#### 11. Distribution of Policy Annual Premium for Fraudulent and Non-Fraudulent Claims
![9](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/0c45c256-a901-43d1-bfe9-b60057d420ff)
- This box plot shows the distribution of policy annual premium for fraudulent and non-fraudulent claims. The annual premiums are slightly higher for fraudulent claims.

#### 12. Distribution of Insured Zip for Fraudulent and Non-Fraudulent Claims
![10](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/05a113ab-b4fa-41f5-92ff-44528dc16828)
- This box plot shows the distribution of insured zip codes for fraudulent and non-fraudulent claims. There is no significant difference between the two groups.

#### 13. Distribution of Authorities Contacted for Fraudulent and Non-Fraudulent Claims
![11](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/543a311d-9ebe-40d0-bbde-344909c0396b)
- This box plot shows the distribution of authorities contacted for fraudulent and non-fraudulent claims. Fraudulent claims tend to involve more authorities.

#### 14. Distribution of Witnesses for Fraudulent and Non-Fraudulent Claims
![12](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/4c2a6cc8-0fb4-4353-8fdd-edb1adc821c1)
- This box plot shows the distribution of the number of witnesses for fraudulent and non-fraudulent claims. Fraudulent claims tend to have more witnesses.

#### 15. Distribution of Number of Vehicles Involved for Fraudulent and Non-Fraudulent Claims
![13](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/544e2998-c407-4269-85a4-075bcb661eb4)
- This box plot shows the distribution of the number of vehicles involved for fraudulent and non-fraudulent claims. Fraudulent claims tend to involve more vehicles.

#### 16. Distribution of Bodily Injuries for Fraudulent and Non-Fraudulent Claims
![14](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/8fbd71a7-d396-49cc-92cc-4bb99f23aace)
- This box plot shows the distribution of bodily injuries for fraudulent and non-fraudulent claims. Fraudulent claims tend to report more bodily injuries.

#### 17. Distribution of Incident Hour of the Day for Fraudulent and Non-Fraudulent Claims
![15](https://github.com/shagoftakhan1605/Insurance_Claim/assets/173546811/9abfd730-5840-44a1-a814-5050fed90ae6)
- This box plot shows the distribution of incident hours for fraudulent and non-fraudulent claims. There is no significant difference between the two groups.

## Usage
To run the analysis:
1. Clone this repository to your local machine.
2. Install the required dependencies.
3. Open the Jupyter notebook `Insurance_Claims_Analysis.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells in the notebook sequentially to perform the analysis.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.
