import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Set plot design
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# 1. GENERATE SYNTHETIC FINANCIAL DATA
def generate_fraud_data(n_samples=50000):
    """Simulates a realistic, heavily imbalanced financial transaction dataset."""
    print("--> Simulating synthetic financial transaction data...")
    np.random.seed(42)
    
    # Generate baseline normal transactions
    df = pd.DataFrame({
        'step': np.random.randint(1, 30, n_samples),                 # Time step / day
        'amount': np.random.exponential(scale=100, size=n_samples),   # Transaction amount
        'old_balance': np.random.uniform(100, 50000, n_samples),
        'device_velocity': np.random.poisson(lam=2, size=n_samples), # Transactions per hour
    })
    df['new_balance'] = df['old_balance'] - df['amount']
    df['is_fraud'] = 0
    
    # Inject Fraud (0.5% of total dataset)
    n_fraud = int(n_samples * 0.005)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    
    # Characteristics of fraudulent transactions: Higher amounts, draining accounts, high velocity
    df.loc[fraud_indices, 'amount'] = np.random.uniform(5000, 20000, n_fraud)
    df.loc[fraud_indices, 'old_balance'] = df.loc[fraud_indices, 'amount'] + np.random.uniform(0, 500, n_fraud)
    df.loc[fraud_indices, 'new_balance'] = df.loc[fraud_indices, 'old_balance'] - df.loc[fraud_indices, 'amount']
    df.loc[fraud_indices, 'device_velocity'] = np.random.poisson(lam=8, size=n_fraud)
    df.loc[fraud_indices, 'is_fraud'] = 1
    
    return df

# Initialize Dataset
data = generate_fraud_data()
print(f"Dataset Shape: {data.shape}")
print(f"Class Distribution:\n{data['is_fraud'].value_counts(normalize=True) * 100}\n")

# 2. DATA PREPROCESSING & SPLITTING
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

# Train/Test Split (Stratified to maintain the 0.5% fraud distribution in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale amounts and balances to prevent large numbers from distorting the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}\n")

# 3. HANDLING IMBALANCE WITH SMOTE
print("--> Applying SMOTE to handle class imbalance...")
# Synthetically generates new fraud samples for the training set only
smote = SMOTE(sampling_strategy=0.1, random_state=42) # Bring fraud class up to 10% of majority class
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"Resampled Training Fraud Distribution:\n{y_train_res.value_counts()}\n")


# 4. MODEL TRAINING (XGBOOST)
print("--> Training XGBoost Classifier...")
# Scale_pos_weight gives more attention to remaining minority class issues
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=3, 
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_res, y_train_res)
print("Model training completed successfully!")

# 5. MODEL EVALUATION
# Predict classes and probabilities
y_pred = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Precision-Recall Curve (The gold standard for fraud evaluation)
precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f'XGBoost (PR-AUC = {pr_auc:.2f})', color='purple', lw=2)
plt.xlabel('Recall (Sensitivity to catch Fraud)')
plt.ylabel('Precision (Accuracy when flagging Fraud)')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR-AUC Score: {pr_auc:.4f}")