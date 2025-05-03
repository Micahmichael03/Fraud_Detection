# Fraud Detection Analysis Notebook Explanation

## Overview
This notebook performs exploratory data analysis and modeling on a financial transaction dataset to detect fraudulent activities. The dataset contains information about various types of financial transactions and their associated details.
 
## Data Loading and Initial Setup
### Cell 1: Importing Required Libraries
```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
```
This cell imports the essential Python libraries for data analysis and visualization:
- pandas: For data manipulation and analysis
- numpy: For numerical computations
- matplotlib and seaborn: For data visualization

### Cell 2: Configuration
```python
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
```
This cell sets up the visualization environment and suppresses warnings for cleaner output.

### Cell 3: Data Loading
```python
df = pd.read_csv("AIML Dataset.csv")
```
Loads the dataset from the CSV file into a pandas DataFrame.

## Data Exploration
### Cell 4: Initial Data Preview
Shows the first row of the dataset, revealing the following columns:
- step: Time step of the transaction
- type: Type of transaction
- amount: Transaction amount
- nameOrig: Origin account identifier
- oldbalanceOrg: Initial balance of origin account
- newbalanceOrig: New balance of origin account
- nameDest: Destination account identifier
- oldbalanceDest: Initial balance of destination account
- newbalanceDest: New balance of destination account
- isFraud: Fraud indicator (1 for fraud, 0 for legitimate)
- isFlaggedFraud: System-flagged fraud indicator

### Cell 5: Dataset Information
The dataset contains:
- 6,362,620 entries
- 11 columns
- Memory usage: 534.0+ MB
- Data types: float64(5), int64(3), object(3)

### Cell 6: Column Names
Lists all column names in the dataset.

### Cell 7: Fraud Distribution
```python
df["isFraud"].value_counts()
```
Shows the distribution of fraudulent vs legitimate transactions:
- Legitimate transactions (0): 6,354,407
- Fraudulent transactions (1): 8,213
This indicates a highly imbalanced dataset with only about 0.13% of transactions being fraudulent.

### Cell 8: Missing Values Check
```python
df.isnull().sum().sum()
```
Confirms that there are no missing values in the dataset.

### Cell 9: Dataset Dimensions
```python
df.shape
```
Shows the dataset dimensions: (6,362,620 rows × 11 columns)

## Key Insights
1. The dataset is large with over 6.3 million transactions
2. There is a significant class imbalance with only 0.13% of transactions being fraudulent
3. The dataset is clean with no missing values
4. The data includes both numerical and categorical features
5. The dataset contains detailed information about both origin and destination accounts

## Next Steps
The notebook continues with more detailed analysis, including:
- Transaction type analysis
- Amount distribution analysis
- Balance changes analysis
- Feature engineering
- Model development and evaluation

## Feature Engineering and Model Development

### Feature Engineering
The notebook creates additional features to help with fraud detection:
- `balanceDiffOrig`: The difference in balance for the origin account
- `balanceDiffDest`: The difference in balance for the destination account

These features help capture the impact of transactions on account balances.

### Model Preparation
1. Feature Selection:
   - Numerical features: 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
   - Categorical features: 'type'
   - Target variable: 'isFraud'
   - Dropped features: 'nameOrig', 'nameDest', 'isFlaggedFraud'

2. Data Preprocessing Pipeline:
   ```python
   preprocessor = ColumnTransformer(
       transformers=[
           ("num", StandardScaler(), numeric_features),
           ("cat", OneHotEncoder(drop="first"), categorical_features)
       ],
       remainder="drop"
   )
   ```
   - Numerical features are standardized using StandardScaler
   - Categorical features are one-hot encoded with the first category dropped

### Model Training
1. Model Selection:
   - Logistic Regression with balanced class weights
   - Maximum iterations: 1000

2. Pipeline Creation:
   ```python
   Pipeline = Pipeline([
       ("prep", preprocessor),
       ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
   ])
   ```

3. Train-Test Split:
   - Test size: 30%
   - Stratified split to maintain fraud ratio

### Model Evaluation
1. Classification Report:
   ```
              precision    recall  f1-score   support
           0       1.00      0.95      0.97   1906322
           1       0.02      0.93      0.04      2464
    accuracy                           0.95   1908786
   ```
   - High accuracy (95%) on legitimate transactions
   - High recall (93%) but very low precision (2%) on fraudulent transactions
   - Overall weighted F1-score of 0.97 due to class imbalance

2. Confusion Matrix:
   ```
   [[1803776,  102546],
    [    166,    2298]]
   ```
   - True Negatives: 1,803,776 (correctly identified legitimate transactions)
   - False Positives: 102,546 (legitimate transactions flagged as fraud)
   - False Negatives: 166 (missed fraud cases)
   - True Positives: 2,298 (correctly identified fraud cases)

## Key Model Insights
1. The model shows strong performance in identifying legitimate transactions
2. High recall for fraud detection (93% of frauds detected)
3. Many false positives, leading to low precision for fraud detection
4. The class imbalance significantly impacts the model's performance metrics
5. The model achieves an overall accuracy of 94.6%

## Potential Improvements
1. Feature engineering to create more discriminative features
2. Try more sophisticated models (e.g., Random Forest, XGBoost)
3. Implement advanced sampling techniques to address class imbalance
4. Consider cost-sensitive learning approaches
5. Explore anomaly detection methods

Note: This explanation covers the initial data exploration phase of the notebook. The complete notebook likely contains more advanced analysis and modeling steps that would be covered in subsequent sections. 