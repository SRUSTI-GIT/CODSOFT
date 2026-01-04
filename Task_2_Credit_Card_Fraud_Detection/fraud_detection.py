import pandas as pd
# Load the dataset
df = pd.read_csv("Task_2_Credit_Card_Fraud_Detection/fraudTest.csv")
# Verify loading
print("Dataset loaded successfully")
print(df.head())
print("\nShape of dataset:", df.shape)
# Dataset information
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
# Check class distribution
print("\nColumn names:")
for col in df.columns:
    print(col)
print("\nTarget column value counts:")
print(df['is_fraud'].value_counts())
# Drop irrelevant / high-cardinality columns
drop_cols = [
    'trans_num', 'first', 'last', 'street', 'city',
    'state', 'zip', 'job', 'dob', 'unix_time'
]
df = df.drop(columns=drop_cols)
print("\nColumns after dropping:")
print(df.columns)
df = df.drop(columns=['Unnamed: 0'])
df = df.drop(columns=['trans_date_trans_time'])

# Load dataset
df = pd.read_csv("Task_2_Credit_Card_Fraud_Detection/fraudTest.csv")

print("Initial columns:")
print(df.columns)

# Target distribution
print("\nTarget column value counts:")
print(df['is_fraud'].value_counts())

# Drop irrelevant columns
drop_cols = [
    'Unnamed: 0',
    'trans_num',
    'first',
    'last',
    'street',
    'city',
    'state',
    'zip',
    'job',
    'dob',
    'unix_time',
    'trans_date_trans_time'
]
# Drop only columns that exist (safety)
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print("\nColumns after dropping:")
print(df.columns)
# Separate features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['merchant', 'category', 'gender']

for col in cat_cols:
    X[col] = le.fit_transform(X[col])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
