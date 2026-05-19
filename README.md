# Titanic Survival Prediction Using Machine Learning

This university project applies machine learning algorithms to predict passenger survival outcomes from the historic Titanic shipwreck using the classic Kaggle dataset.

## 📌 Project Overview
The goal of this project is to build a predictive model that answers the question: “What sorts of people were more likely to survive?” using passenger data (i.e., name, age, gender, socio-economic class, etc.).

## 📊 Dataset Description
The dataset contains demographics and traveling information for 891 passengers in the training set. Key features include:
* **PassengerId:** Unique ID for each passenger.
* **Survived:** Survival label (0 = No, 1 = Yes).
* **Pclass:** Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
* **Sex:** Passenger gender.
* **Age:** Passenger age in years.
* **SibSp:** Number of siblings or spouses aboard.
* **Parch:** Number of parents or children aboard.
* **Ticket:** Ticket number.
* **Fare:** Passenger fare.
* **Cabin:** Cabin number.
* **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## 🛠️ Project Workflow
1. **Data Cleaning:** Handled missing data in `Age`, `Cabin`, and `Embarked` columns.
2. **Feature Engineering:** Extracted passenger titles, created family size metrics, and encoded categorical variables.
3. **Exploratory Data Analysis (EDA):** Visualized survival correlations based on gender, class, and age.
4. **Model Training:** Built and evaluated multiple classifiers (e.g., Logistic Regression, Decision Tree, SVM).
5. **Hyperparameter Tuning:** Optimized the best-performing model using Grid Search.

## 📈 Results
* **Best Model:** Decision Tree
* **Validation Accuracy:** 82.30%
