import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load the data
data = pd.read_csv("/content/emails.csv")
data = data.drop('Email No.', axis=1)

# Display basic information about the dataset
print(data.shape)
print(data.describe())
print(data.info())
print(data['Prediction'].value_counts())

# Split features and target variable
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# K-Nearest Neighbors Classifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)

# Predict on the test set
y_pred_knn = neigh.predict(X_test)

# Print accuracy scores
print(f"KNN Training Accuracy: {neigh.score(X_train, y_train)}")
print(f"KNN Testing Accuracy: {neigh.score(X_test, y_test)}")

# Confusion Matrix and Classification Report for KNN
print("KNN Confusion Matrix:")
cm_knn = confusion_matrix(y_test, y_pred_knn)
ConfusionMatrixDisplay(confusion_matrix=cm_knn).plot()
plt.title("KNN Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred_knn))
print("KNN Accuracy Score:", accuracy_score(y_test, y_pred_knn))
print("KNN Precision Score:", precision_score(y_test, y_pred_knn, average='weighted'))
print("KNN Recall Score:", recall_score(y_test, y_pred_knn, average='weighted'))
print("KNN Error Rate:", 1 - accuracy_score(y_test, y_pred_knn))

# Support Vector Machine Classifier
SVM = SVC(gamma='auto')
SVM.fit(X_train, y_train)

# Predict on the test set using SVM
y_pred_svm = SVM.predict(X_test)

# Print accuracy scores for SVM
print(f"SVM Training Accuracy: {SVM.score(X_train, y_train)}")
print(f"SVM Testing Accuracy: {SVM.score(X_test, y_test)}")

# Confusion Matrix and Classification Report for SVM
print("SVM Confusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
ConfusionMatrixDisplay(confusion_matrix=cm_svm).plot()
plt.title("SVM Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred_svm))
print("SVM Accuracy Score:", accuracy_score(y_test, y_pred_svm))
print("SVM Precision Score:", precision_score(y_test, y_pred_svm, average='weighted'))
print("SVM Recall Score:", recall_score(y_test, y_pred_svm, average='weighted'))
print("SVM Error Rate:", 1 - accuracy_score(y_test, y_pred_svm))