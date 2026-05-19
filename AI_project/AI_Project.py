import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

print("Train Head:")
print(train_df.head())
print("\nTest Head:")
print(test_df.head())
print("\nTrain Info:")
print(train_df.info())
print("\nTest Info:")
print(test_df.info())

# -------------------------------------------------------------------
# 2. Data Visualization
# -------------------------------------------------------------------
print("\nGenerating Data Visualizations...")

# 1. Survival by Sex
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.title('Survival by Sex')
plt.show()

# 2. Survival by Pclass
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=train_df)
plt.title('Survival by Passenger Class')
plt.show()

# 3. Age Distribution by Survival
plt.figure(figsize=(8, 5))
sns.histplot(data=train_df, x='Age', hue='Survived', multiple='stack', kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# 4. Correlation Matrix (Numeric features only)
plt.figure(figsize=(10, 6.5))
numeric_train_df = train_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# -------------------------------------------------------------------
# 3. Preprocessing
# -------------------------------------------------------------------
target = train_df['Survived']
train_features = train_df.drop('Survived', axis=1)

# Drop columns that are irrelevant or have too many missing values
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin'] 
train_features = train_features.drop(columns=columns_to_drop)

# Handle missing values
train_features['Age'] = train_features['Age'].fillna(train_features['Age'].median())
train_features['Fare'] = train_features['Fare'].fillna(train_features['Fare'].median())
train_features['Embarked'] = train_features['Embarked'].fillna(train_features['Embarked'].mode()[0])

# Visualize Age and Fare BEFORE IQR
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=train_features, y='Age', color='lightblue')
plt.title('Age Before IQR')
plt.subplot(1, 2, 2)
sns.boxplot(data=train_features, y='Fare', color='lightgreen')
plt.title('Fare Before IQR')
plt.tight_layout()
plt.show()

# Handle outliers (Capping using IQR method)
for col in ['Age', 'Fare']:
    Q1 = train_features[col].quantile(0.25)
    Q3 = train_features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train_features[col] = np.clip(train_features[col], lower_bound, upper_bound)

# Visualize Age and Fare AFTER IQR
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=train_features, y='Age', color='lightblue')
plt.title('Age After IQR')
plt.subplot(1, 2, 2)
sns.boxplot(data=train_features, y='Fare', color='lightgreen')
plt.title('Fare After IQR')
plt.tight_layout()
plt.show()

# Encode categorical features
train_features['Sex'] = train_features['Sex'].map({'male': 0, 'female': 1})
train_features = pd.get_dummies(train_features, columns=['Embarked'], drop_first=True)
train_preprocessed = train_features
train_preprocessed['Survived'] = target.values

print("\n--- After Preprocessing ---")
train_preprocessed.info()

# -------------------------------------------------------------------
# 4. Model Training and Hyperparameter Tuning
# -------------------------------------------------------------------
X = train_preprocessed.drop('Survived', axis=1)
y = train_preprocessed['Survived']

# Split into training and test sets to evaluate models locally
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42) # change  test size


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_processed = test_df.drop(columns=columns_to_drop)
test_processed['Age'] = test_processed['Age'].fillna(train_df['Age'].median()) 
test_processed['Fare'] = test_processed['Fare'].fillna(train_df['Fare'].median())
test_processed['Embarked'] = test_processed['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_processed['Sex'] = test_processed['Sex'].map({'male': 0, 'female': 1})
test_processed = pd.get_dummies(test_processed, columns=['Embarked'], drop_first=True)

#Forces test columns to EXACTLY match training columns
test_processed = test_processed.reindex(columns=X.columns, fill_value=0)
test_scaled = scaler.transform(test_processed)

# data frame
results = []

# 1. Logistic Regression
print("\n--- 1. Logistic Regression ---")
lr_grid = GridSearchCV(LogisticRegression(random_state=42), {'C':[0.01], 'max_iter': [500, 1000]}, cv=5, scoring='accuracy')
lr_grid.fit(X_train_scaled, y_train)
lr_pred = lr_grid.predict(X_test_scaled)
print(classification_report(y_test, lr_pred, target_names=['Not Survived', 'Survived']))
results.append({"Model": "Logistic Regression",
                "Best Params": lr_grid.best_params_ ,
                "CV Mean Accuracy": lr_grid.best_score_,
                "Validation Accuracy": accuracy_score(y_test, lr_pred)})

# 2. Decision Tree
print("\n--- 2. Decision Tree ---")
dt_grid = GridSearchCV(DecisionTreeClassifier(criterion='entropy',random_state=42), {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}, cv=5, scoring='accuracy')
dt_grid.fit(X_train, y_train)
dt_pred = dt_grid.predict(X_test)
print(classification_report(y_test, dt_pred, target_names=['Not Survived', 'Survived']))
results.append({"Model": "Decision Tree",
                "Best Params": dt_grid.best_params_,
                "CV Mean Accuracy": dt_grid.best_score_,
                "Validation Accuracy": accuracy_score(y_test, dt_pred)})

# 3. Support Vector Machine (SVM)
print("\n--- 3. Support Vector Machine (SVM) ---")
svm_grid = GridSearchCV(SVC(random_state=42), {'C': [0.005], 'kernel': ['linear']}, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
svm_pred = svm_grid.predict(X_test_scaled)
print(classification_report(y_test, svm_pred, target_names=['Not Survived', 'Survived']))
results.append({"Model": "SVM",
                "Best Params": svm_grid.best_params_,
                "CV Mean Accuracy": svm_grid.best_score_,
                "Validation Accuracy": accuracy_score(y_test, svm_pred)})

# # -------------------------------------------------------------------
# # 4. Predictions on the Testing Set 
# # -------------------------------------------------------------------
print("\n--- Generating Predictions on Titanic_test.csv ---")

# Predict on test data using all three tuned models
lr_test_pred = lr_grid.predict(test_scaled)
dt_test_pred = dt_grid.predict(test_scaled)
svm_test_pred = svm_grid.predict(test_scaled)

lr_results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': lr_test_pred})
dt_results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': dt_test_pred})
svm_results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': svm_test_pred})

print("\n--- Logistic Regression Test Predictions ---")
print(lr_results)

print("\n--- Decision Tree Test Predictions ---")
print(dt_results)

print("\n--- SVM Test Predictions ---")
print(svm_results)

# Print summary of results to terminal
summary_df = pd.DataFrame(results)
print("\n--- Model Summary ---\n")
print(summary_df.to_string())

# Save the best model (Logistic Regression) and the scaler for the GUI
print("\n--- Saving Model and Scaler ---")
joblib.dump(lr_grid.best_estimator_, 'lr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Saved 'lr_model.pkl' and 'scaler.pkl' successfully.")