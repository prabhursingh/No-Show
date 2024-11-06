Dataset appears to be related to healthcare appointments, potentially for predicting no-shows.

Research Question

Which demographic factors affect biases in Machine Learning prediction models for no-show in healthcare appointments?

Which techniques can be used to mitigate these biases?
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math



import pandas as pd
df = pd.read_csv(r"D:\Daily file\Semester 3\Group Project\KaggleV2-May-.csv")



# General Data Overview
print("\nData Overview")
print("Shape of dataset:", df.shape)
print("Columns:", df.columns)
print(df.head())
# Data Information and Summary

print("\nDataset Information")
df.info()
print("\nSummary Statistics")
print(df.describe())
# Check for Missing Values

print("\nMissing Values Count")
print(df.isnull().sum())
# Unique Values and Data Types

for column in df.columns:
    print(f"\nUnique values in {column}: {df[column].nunique()}")
    print(df[column].unique())
df.dtypes
### convert data type
df["Neighbourhood"] = df["Neighbourhood"].astype("category")
df['Scholarship'] = df['Scholarship'].astype(bool)
df['Hypertension'] = df['Hypertension'].astype(bool)
df['Diabetes'] = df['Diabetes'].astype(bool)
df['Alcoholism'] = df['Alcoholism'].astype(bool)
df['Handicap'] = df['Handicap'].astype(bool)
df['SMS_received'] = df['SMS_received'].astype(bool)

df.dtypes
df['Gender']= df['Gender'].map({'M': 1, 'F': 0})
# Define age bins and corresponding labels for Medium Level classification
bins = [-1, 14, 24, 44, 64, 105]
labels = ['under 15', '15-24', '25-44', '45-64', '65+']

# Apply the binning and labeling to the 'Age' column
df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Display the first few rows to check the result
df.head()

# Univariate Analysis - Categorical Features

categorical_cols = df.select_dtypes(include=['category']).columns
for col in categorical_cols:
    # Get the value counts for the column
    value_counts = df[col].value_counts()
    print(f"\nValue counts for {col}:")
    print(value_counts)
    
    # Plot the value counts, limiting to the top 10 categories for faster plotting
    plt.figure(figsize=(10, 5))
    sns.countplot(x=col, data=df, order=value_counts.index[:10])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
# Univariate Analysis - Numerical Features
# Select numerical columns from the DataFrame
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# List of columns to exclude from plotting
exclude_cols = ['PatientId', 'AppointmentID']

# Filter out the columns you want to exclude
filtered_numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

# Set up the figure for subplots
num_plots = len(filtered_numerical_cols)
cols = 3  # Number of columns for the subplot grid
rows = math.ceil(num_plots / cols)  # Calculate the number of rows needed

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Iterate through each filtered numerical column and plot
for i, col in enumerate(filtered_numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Bivariate Analysis - Target vs Features

if 'No-show' in df.columns:
    target = 'No-show'
    features = ['Gender', 'age_group', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']

    # Convert features to categorical where necessary
    for col in features:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Iterate through each feature
    for col in features:
        if col in df.columns:
            # Print unique values for each feature to check for issues
            print(f"Feature: {col}, Unique values: {df[col].unique()}")
            print(f"Target: {target}, Unique values: {df[target].unique()}")

            # Drop rows with NaN values in target and feature columns
            df_subset = df.dropna(subset=[target, col])

            # Proceed to plot only if data is available
            if not df_subset.empty:
                plt.figure(figsize=(10, 5))
                sns.countplot(x=target, hue=col, data=df_subset)  # Removed order to check the actual categories
                plt.title(f"Relationship between No-show and {col}")
                plt.xticks(rotation=45)
                plt.show()

df.dtypes
# Define categorical columns, including 'No-show'
categorical_cols = ['Gender', 'age_group', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']

# Label encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        # Only apply encoding if the column is not numeric
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = le.fit_transform(df[col])

# Select all numerical columns, including encoded categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'bool', 'float64']).columns
df.dtypes
# Correlation Matrix for Numerical Features

# Define categorical columns, including 'No-show'
categorical_cols = ['Gender', 'age_group', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'No-show']

# Label encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        # Only apply encoding if the column is not numeric
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = le.fit_transform(df[col])

# Explicitly check if 'No-show' is correctly encoded
print(df['No-show'].head())  # Check first few values to verify encoding

# Select all numerical columns, including encoded categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'bool', 'float64']).columns

# Make sure 'No-show' is in the numerical columns list
if 'No-show' not in numerical_cols:
    numerical_cols = numerical_cols.append(pd.Index(['No-show']))

# Correlation Matrix for Numerical Features including encoded categorical columns and 'No-show'
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix including Encoded Categorical Features and 'No-show'")
plt.show()

# Plot a combined box plot
df_melted = df[['Age', 'Hypertension', 'Diabetes']].melt(var_name='Feature', value_name='Value')
plt.figure(figsize=(10, 5))
sns.boxplot(x='Feature', y='Value', data=df_melted)
plt.title("Box Plot for Age, Hypertension, and Diabetes")
plt.show()
# Feature Engineering and Transformation
# Creating new features, encoding categorical variables, etc.
# Example: One-hot encoding categorical variables
if 'df' in locals() or 'df' in globals():
    X = df[['Gender', 'age_group', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received']]
else:
    raise NameError("DataFrame 'df' is not defined. Please run the previous cells to define 'df'.")
encoded_df = pd.get_dummies(X, drop_first=True)
print("\nDataset after Encoding")
print(encoded_df.head())

# Summary Insights
print("\nSummary Insights")
print("1. Description of trends seen in visualizations.")
print("2. Mention any noticeable relationships or outliers.")
print("3. Data quality issues or next steps for modeling.")

X.info
X = df[['Gender','age_group','Scholarship','Hypertension','Diabetes','Alcoholism','Handicap','SMS_received']]
# Indicate that the column is an ordered categorical feature
categ = ["under 15","15-24","25-44" "45-64" "65+"]
X["age_group"] = pd.Categorical(X["age_group"], categories=categ, ordered=True)
print(X.dtypes)
# Get the factors and replace with numbers
labels, unique = pd.factorize(X["age_group"], sort=True)
print(labels, unique)
# We can replace the column with the labels
X["age_group"] = labels
X
X.info
#If age is used as feature, following normalization process will be required.

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
numerical_features = ['age_group']
X[numerical_features] = (min_max_scaler.fit_transform(X[numerical_features]))
### select target for model training
y = df['No-show']
y.shape
## Machine Learning Model Development
### Dataset is split into training and testing set. Several classification models will be trained on training data and its performance will be measured on testing data. After comparing the performance metrics, one classficiation model will be chosen.
Verify scikit-learn Installation
!pip install -U scikit-learn


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
print("Imports successful")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
X_test.shape
print(X_test.shape, X_train.shape)
### Logistic Regression
log_reg = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
log_reg.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
#import classes for training different classifciation algorithm
### Create performance metrics dataframe to store the metrics of trained models <br> A function for both training and testing is executed, so that the ML models can be trained easily
# Initialize an empty dataframe to store the performance metrics
performance_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
def evaluate_model(model_name, model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' for handling imbalanced classes
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Append the results to the performance dataframe
    performance_df.loc[len(performance_df)] = [model_name, accuracy, precision, recall, f1]
    return performance_df
### Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
evaluate_model('Logistic Regression', log_reg, X_train, X_test, y_train, y_test)
### Decision Tree (2 variants)
DT_gini = DecisionTreeClassifier( criterion='gini',max_depth = 5)
evaluate_model('DT_gini', DT_gini)
DT_entropy = DecisionTreeClassifier( criterion='entropy', max_depth = 5)
evaluate_model('DT_entropy', DT_entropy)
### Random Forest (2 variants)
RF_gini = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=5)
evaluate_model('RF_gini', RF_gini)
RF_entropy = RandomForestClassifier(n_estimators=30, criterion='entropy', max_depth=5)
evaluate_model('RF_entropy', RF_entropy)
### KNN (2 variants)
KNN_5neighbours = KNeighborsClassifier(n_neighbors=5)
evaluate_model('KNN_5neighbours',KNN_5neighbours)
KNN_3neighbours = KNeighborsClassifier(n_neighbors=3)
evaluate_model('KNN_3neighbours',KNN_3neighbours)
# Initialize Gradient Boosting classifier
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
evaluate_model('Gradient Boosting', gradient_boosting)
# Initialize Bagging classifier with Decision Trees as base estimators
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
evaluate_model('Bagging', bagging)
# Initialize AdaBoost classifier with Decision Trees as weak learners
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=1.0)
evaluate_model('Adaptive Boosting', adaboost)
# Initialize a basic Multi-Layer Perceptron (Neural Network)
neural_network = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, activation='relu', solver='adam')
evaluate_model('Neural Network', neural_network)
### Logistic regression model will be used in further bias identification process.
## Bias Identification
### 4 fairness metrics: Demographic Parity Difference, Equalized Odds Difference, Disparate Impact, Negative Rate Difference will be used. <br> These metrics will be measured for each feature to detect that this feature has bias on the logistic regression model.
### first, the required libraries for fairness metrics will be installed
pip install fairlearn
pip install aif360
pip install aif360[inFairness]
### Import required modules
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from sklearn.metrics import accuracy_score, confusion_matrix
### Select a feature to test as demographic data and integrated into the dataframe <br> The dataframe has actual target(no-show), predicted target, and this feature as demographic.
demographic_data = df['Gender']
y_true = y
y_pred_all = log_reg.predict(X)
# Create a dataframe for convenience
df_results = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred_all,
    'demographic': demographic_data
})

df_results.head()
# Compute accuracy overall
accuracy = accuracy_score(df_results['y_true'], df_results['y_pred'])
print("Overall accuracy:", accuracy)
### 2 Metrics Demographic Parity Difference, Equalized Odds Difference will be measured on that feature as follows.
# Fairness metrics: Demographic Parity and Equalized Odds
demographic_parity_diff = demographic_parity_difference(df_results['y_true'], df_results['y_pred'], sensitive_features=df_results['demographic'])
equalized_odds_diff = equalized_odds_difference(df_results['y_true'], df_results['y_pred'], sensitive_features=df_results['demographic'])

print(f"Demographic Parity Difference: {demographic_parity_diff}")
print(f"Equalized Odds Difference: {equalized_odds_diff}")

### Disparate Impact metrics is measured as follows.
# Additional bias metrics using AIF360
# Convert the dataframe to a BinaryLabelDataset object
dataset = BinaryLabelDataset(
    df=df_results,
    label_names=['y_true'],
    protected_attribute_names=['demographic']
)

# Compute metrics
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{'demographic': 1}], unprivileged_groups=[{'demographic': 0}])

# Disparate impact
disparate_impact = metric.disparate_impact()
print(f"Disparate Impact: {disparate_impact}")
### False Negative Rate Difference metrics is measured as follows.
# Calculate False Negative Rate (FNR) for privileged and unprivileged groups manually
# First, separate the privileged and unprivileged groups
privileged_group = df_results[df_results['demographic'] == 1]  # Assuming '1' represents privileged (e.g., Male)
unprivileged_group = df_results[df_results['demographic'] == 0]  # Assuming '0' represents unprivileged (e.g., Female)

# False Negative Rate (FNR) for Privileged Group
tn_priv, fp_priv, fn_priv, tp_priv = confusion_matrix(privileged_group['y_true'], privileged_group['y_pred']).ravel()
fnr_privileged = fn_priv / (fn_priv + tp_priv) if (fn_priv + tp_priv) > 0 else 0

# False Negative Rate (FNR) for Unprivileged Group
tn_unpriv, fp_unpriv, fn_unpriv, tp_unpriv = confusion_matrix(unprivileged_group['y_true'], unprivileged_group['y_pred']).ravel()
fnr_unprivileged = fn_unpriv / (fn_unpriv + tp_unpriv) if (fn_unpriv + tp_unpriv) > 0 else 0

# Difference in False Negative Rates
fnr_difference = fnr_privileged - fnr_unprivileged

print(f"False Negative Rate for Privileged Group: {fnr_privileged}")
print(f"False Negative Rate for Unprivileged Group: {fnr_unprivileged}")
print(f"False Negative Rate Difference: {fnr_difference}")
### According to the results of 4 metrics, the feature 'Gender' has no bias on the model
### A function is structured and executed to calculate these 4 metrics on each feature easily.
def calculate_bias(y_true, y_pred, demographic_data):
    df_results = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred_all,
    'demographic': demographic_data
    })
    accuracy = accuracy_score(df_results['y_true'], df_results['y_pred'])
    print("Overall accuracy:", accuracy)
    demographic_parity_diff = demographic_parity_difference(df_results['y_true'], df_results['y_pred'], sensitive_features=df_results['demographic'])
    equalized_odds_diff = equalized_odds_difference(df_results['y_true'], df_results['y_pred'], sensitive_features=df_results['demographic'])

    print(f"Demographic Parity Difference: {demographic_parity_diff}")
    print(f"Equalized Odds Difference: {equalized_odds_diff}")

    # Additional bias metrics using AIF360
    # Convert the dataframe to a BinaryLabelDataset object
    dataset = BinaryLabelDataset(
    df=df_results,
    label_names=['y_true'],
    protected_attribute_names=['demographic']
    )

# Compute metrics
    metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{'demographic': 1}], unprivileged_groups=[{'demographic': 0}])

# Disparate impact
    disparate_impact = metric.disparate_impact()
    print(f"Disparate Impact: {disparate_impact}")

    # Calculate False Negative Rate (FNR) for privileged and unprivileged groups manually
    # First, separate the privileged and unprivileged groups
    privileged_group = df_results[df_results['demographic'] == 1]  # Assuming '1' represents privileged (e.g., Male)
    unprivileged_group = df_results[df_results['demographic'] == 0]  # Assuming '0' represents unprivileged (e.g., Female)

    # False Negative Rate (FNR) for Privileged Group
    tn_priv, fp_priv, fn_priv, tp_priv = confusion_matrix(privileged_group['y_true'], privileged_group['y_pred']).ravel()
    fnr_privileged = fn_priv / (fn_priv + tp_priv) if (fn_priv + tp_priv) > 0 else 0

    # False Negative Rate (FNR) for Unprivileged Group
    tn_unpriv, fp_unpriv, fn_unpriv, tp_unpriv = confusion_matrix(unprivileged_group['y_true'], unprivileged_group['y_pred']).ravel()
    fnr_unprivileged = fn_unpriv / (fn_unpriv + tp_unpriv) if (fn_unpriv + tp_unpriv) > 0 else 0

    # Difference in False Negative Rates
    fnr_difference = fnr_privileged - fnr_unprivileged

    print(f"False Negative Rate for Privileged Group: {fnr_privileged}")
    print(f"False Negative Rate for Unprivileged Group: {fnr_unprivileged}")
    print(f"False Negative Rate Difference: {fnr_difference}")
### Then, all binary features are used to measure the metrics, one by one.
demographic_data = df['Scholarship']
calculate_bias(y_true, y_pred, demographic_data)
demographic_data = df['Hypertension']
calculate_bias(y_true, y_pred, demographic_data)
demographic_data = df['Diabetes']
calculate_bias(y_true, y_pred, demographic_data)
demographic_data = df['Alcoholism']
calculate_bias(y_true, y_pred, demographic_data)
demographic_data = df['Handicap']
calculate_bias(y_true, y_pred, demographic_data)
demographic_data = df['SMS_received']
calculate_bias(y_true, y_pred, demographic_data)
### age_group is not binary, and it is ordnial feature having 3 categories. So, some modification needs to be done to measure the metrics.
print("Unique age_group values:", df['age_group'].unique())

df['age_group'] = df['age_group'].replace({
    0: 'under 15', 
    1: '15-24', 
    2: '25-44', 
    3: '45-64', 
    4: '65+'
})

demographic_data = df['age_group']
df_results = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred_all,
    'demographic_age_group': demographic_data
})
df_results
### Calculate Demographic Parity Difference, Equalized Odds Difference as follows. 
# Convert all demographic_age_group values to string type
df_results['demographic_age_group'] = df_results['demographic_age_group'].astype(str)

###Verify Column Consistency
print("Unique values in demographic_age_group:", df_results['demographic_age_group'].unique())

### Calculate Demographic Parity Difference, Equalized Odds Difference as follows. 
accuracy = accuracy_score(df_results['y_true'], df_results['y_pred'])
print("Overall accuracy:", accuracy)
demographic_parity_diff = demographic_parity_difference(df_results['y_true'], df_results['y_pred'], sensitive_features=df_results['demographic_age_group'])
equalized_odds_diff = equalized_odds_difference(df_results['y_true'], df_results['y_pred'], sensitive_features=df_results['demographic_age_group'])

print(f"Demographic Parity Difference: {demographic_parity_diff}")
print(f"Equalized Odds Difference: {equalized_odds_diff}")
### For Disparate Impact, age_group has to be numerical. So, ['under 15', '15-24', '25-44', '45-64', '65+']
print(df_results['demographic_age_group'].unique())
# Indicate that the column is an ordered categorical feature
categ = ['under 15', '15-24', '25-44', '45-64', '65+']
df_results['demographic_age_group'] = pd.Categorical(df_results['demographic_age_group'], categories=categ, ordered=True)
print(df_results.dtypes)
labels, unique = pd.factorize(df_results["demographic_age_group"], sort=True)
print(labels, unique)
df_results["demographic_age_group"] = labels
df_results
df_results.isnull().sum()
### To measure the Disparate Impact and False Negative Rate Difference, it has to be 2 categories. So, 3 categories of age_group are transformed into 2 categories only: privilieged group and unprivileged group. Then, the metrics are measured.
# Convert the dataframe to a BinaryLabelDataset object
dataset = BinaryLabelDataset(
    df=df_results,
    label_names=['y_true'],  # Ensure 'y_true' is the correct label name in your DataFrame
    protected_attribute_names=['demographic_age_group']  # Ensure this is a list
)

# Print unique values of age_group for verification
print("Unique age_group values:", df_results['demographic_age_group'].unique())

# Define privileged and unprivileged groups
privileged_groups = [{'demographic_age_group': 0}]  # Change based on your understanding of the data
unprivileged_groups = [{'demographic_age_group': 1 }, {'demographic_age_group': 2}]  # Include all relevant unprivileged groups

# Compute metrics
metric = BinaryLabelDatasetMetric(dataset, privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

# Calculate disparate impact
disparate_impact = metric.disparate_impact()

print(f"Disparate Impact: {disparate_impact}")
# False Negative Rate (FNR) for Privileged Group
tn_priv, fp_priv, fn_priv, tp_priv = confusion_matrix(privileged_group['y_true'], privileged_group['y_pred']).ravel()
fnr_privileged = fn_priv / (fn_priv + tp_priv) if (fn_priv + tp_priv) > 0 else 0

# False Negative Rate (FNR) for Unprivileged Group
tn_unpriv, fp_unpriv, fn_unpriv, tp_unpriv = confusion_matrix(unprivileged_group['y_true'], unprivileged_group['y_pred']).ravel()
fnr_unprivileged = fn_unpriv / (fn_unpriv + tp_unpriv) if (fn_unpriv + tp_unpriv) > 0 else 0

# Difference in False Negative Rates
fnr_difference = fnr_privileged - fnr_unprivileged
print(f"False Negative Rate for Privileged Group: {fnr_privileged}")
print(f"False Negative Rate for Unprivileged Group: {fnr_unprivileged}")
print(f"False Negative Rate Difference: {fnr_difference}")
### SHAP visualization is done to know the impact of each feature on the model.(It is optional)
import shap
# Create a SHAP explainer for linear models
explainer = shap.LinearExplainer(log_reg, X_train)  # For linear models

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)
shap_values.shape
feat_names=list(X.columns)
feat_names
X_train = np.array(X_train, dtype=np.float64)
X_test = np.array(X_test, dtype=np.float64)

# Create SHAP explainer for linear models
explainer = shap.LinearExplainer(log_reg, X_train)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)
### Visulization 
shap.summary_plot(shap_values, X_test, feature_names=feat_names)
This SHAP summary plot shows the impact of features on the model’s predictions:

Y-axis: Lists features, ordered by their overall impact.
X-axis (SHAP values): Indicates impact direction; positive values increase predictions, negative values decrease them.
Color: Represents feature values (red = high, blue = low).
Key insights:

SMS_received: Lower values reduce predictions.
Scholarship: Higher values increase predictions.
Gender, Alcoholism, Handicap: Vary in impact, depending on value.