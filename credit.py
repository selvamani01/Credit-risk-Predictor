import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")
df = df[df['loan_status'].notna()]
df.dropna(inplace=True)

# Encode categorical columns and store encoders
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # save the fitted encoder


# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Store model scores and predictions
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print(cm)
    print(classification_report(y_test, preds))
    
    # Save for visual comparison
    results[name] = {
        "model": model,
        "accuracy": acc,
        "conf_matrix": cm
    }

# ----------- Visualization ----------
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]["accuracy"] for m in model_names]
sns.barplot(x=model_names, y=accuracies)
plt.ylabel("Accuracy Score")
plt.title("Model Comparison")
plt.ylim(0, 1)
plt.show()
# Manually define categorical columns (original string-based ones)
categorical_cols = [
    'person_home_ownership', 'loan_intent',
    'loan_grade', 'cb_person_default_on_file'
]



# ----------- Predict From User Input ----------
print("\nEnter user details for prediction:")
user_data = {}
for col in X.columns:
    val = input(f"{col}: ")
    if col in categorical_cols:
        val = encoders[col].transform([val])[0]
    else:
        val = float(val)
    user_data[col] = val


user_df = pd.DataFrame([user_data])
user_df_scaled = scaler.transform(user_df)
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]
user_pred = best_model.predict(user_df_scaled)[0]

print(f"\nUsing {best_model_name} → Prediction: {'High Risk' if user_pred == 1 else 'Low Risk'}")

# ----------- Save Model, Scaler, Encoders ----------
import pickle

# Save the best model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("\n✅ model.pkl, scaler.pkl, and encoders.pkl saved successfully.")
