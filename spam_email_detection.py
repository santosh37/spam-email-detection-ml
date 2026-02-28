# ============================================================
# SPAM EMAIL DETECTION SYSTEM USING MACHINE LEARNING
# ============================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# 2. Load Dataset
# ============================================================

# Download spam.csv from Kaggle (SMS Spam Collection)
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print("Dataset Loaded Successfully")
print(data.head())


# ============================================================
# 3. Convert Labels to Numeric
# ============================================================

data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# ============================================================
# 4. Feature Extraction (TF-IDF)
# ============================================================

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(data['message'])
y = data['label']


# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 6. Model Training
# ============================================================

# Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Support Vector Machine
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


# ============================================================
# 7. Evaluation Function
# ============================================================

def evaluate_model(name, y_test, y_pred):
    print("\nModel:", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("---------------------------------------")


# ============================================================
# 8. Evaluate Models
# ============================================================

evaluate_model("Naive Bayes", y_test, y_pred_nb)
evaluate_model("SVM", y_test, y_pred_svm)


# ============================================================
# 9. Cross Validation
# ============================================================

print("\nCross Validation (5-Fold):")

cv_nb = cross_val_score(nb, X, y, cv=5).mean()
cv_svm = cross_val_score(svm, X, y, cv=5).mean()

print("Naive Bayes CV Accuracy:", cv_nb)
print("SVM CV Accuracy:", cv_svm)


# ============================================================
# 10. Accuracy Comparison Graph
# ============================================================

acc_nb = accuracy_score(y_test, y_pred_nb)
acc_svm = accuracy_score(y_test, y_pred_svm)

models = ["Naive Bayes", "SVM"]
accuracies = [acc_nb, acc_svm]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()


# ============================================================
# 11. Confusion Matrix Heatmap (SVM)
# ============================================================

cm = confusion_matrix(y_test, y_pred_svm)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 12. Manual Test Prediction
# ============================================================

print("\nManual Testing Example:")

sample_email = ["Congratulations! You have won a free lottery ticket. Call now!"]
sample_vector = vectorizer.transform(sample_email)
prediction = svm.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction: Spam")
else:
    print("Prediction: Not Spam")

# ============================================================
# 13. ROC Curve
# ============================================================

from sklearn.metrics import roc_curve, auc

y_prob_svm = svm.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_prob_svm)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve - SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("ROC AUC Score:", roc_auc)


# ============================================================
# 14. Save Model and Vectorizer
# ============================================================

import joblib

joblib.dump(svm, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and Vectorizer saved successfully.")
