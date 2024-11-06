'''
Assignment: Machine Learning for Genomic Data. Task: Apply machine learning algorithms, such as random
forests or support vector machines, to classify genomic data based on specific features or markers. Deliverable: A
comprehensive analysis report presenting the classification results, model performance evaluation, and insights
into the predictive features.

Viva Questions - 
1) Explain SVM
2) Explain Random Forrest
'''

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load and process the dataset
metabric_df = pd.read_csv("genomic_data.csv")
metabric_df = metabric_df.set_index('patient_id')
df_expression = metabric_df.iloc[:, 30:519].join(metabric_df['overall_survival'], how='inner')

# Check for missing values
print(df_expression.isnull().sum())

# Dictionary to store F1 and accuracy scores of each model
metrics_summary = {"Model": [], "F1 Score": [], "Accuracy": []}

# Function to evaluate and display results
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics_summary["Model"].append(model_name)
    metrics_summary["F1 Score"].append(f1)
    metrics_summary["Accuracy"].append(accuracy)

    print(f"\n=== {model_name} ===")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    if y_proba is not None:
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}")

def main(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features (optional for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define classifiers to be evaluated
    models = {
        'Random Forest': RandomForestClassifier(),
        'Support Vector Classifier': SVC(probability=True),
    }

    for name, model in models.items():
        evaluate_model(model, X_train, X_test, y_train, y_test, name)

# Load your dataset and specify the target column
if __name__ == "__main__":
    df = df_expression  # Example input dataframe
    target_column = 'overall_survival'  # Replace with the actual target column name
    main(df, target_column)


