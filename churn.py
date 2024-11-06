import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("/content/Churn_Modelling.csv")

# Check for null values
print(df.isnull().sum())

# Describe the dataset
print(df.describe())
print(df.info())
print(df.columns)

# Drop unnecessary columns
df = df.drop(['Surname', 'CustomerId', 'RowNumber'], axis=1)

# Prepare features and labels
X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
states = pd.get_dummies(df['Geography'], drop_first=True)
gender = pd.get_dummies(df['Gender'], drop_first=True)

# Concatenate the one-hot encoded variables with the original dataframe
X = pd.concat([X, gender, states], axis=1)
y = df['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the ANN model
classifier = Sequential()
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=X_train.shape[1])) 
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# Compile the model
classifier.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
classifier.summary()

# Fit the ANN to the training dataset
classifier.fit(X_train, y_train, batch_size=10, epochs=50)

# Make predictions on the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))