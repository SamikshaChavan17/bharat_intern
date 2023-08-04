#!/usr/bin/env python
# coding: utf-8

# In[ ]:


BHARAT INTERN 
SAMIKSHA CHAVAN 
K.K.WAGH COLLEGE 


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset (make sure you have the CSV file)
data = pd.read_csv(r"C:\Users\samik\OneDrive\Desktop\Project\ds\test.csv")

# Data preprocessing
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into numerical
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Select features and target
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Train the model
clf.fit(X_train_imputed, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

classification_rep = classification_report(y_test, y_pred)
print(classification_rep)

# Feature importance analysis
feature_importances = clf.feature_importances_
feature_names = X.columns

# Create a bar plot of feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# In[ ]:





# In[ ]:




