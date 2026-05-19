import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to load the model and scaler
try:
    model = joblib.load('lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    model = None
    scaler = None

# Try to load the dataset for visualizations
try:
    train_df = pd.read_csv("Titanic_train.csv")
except FileNotFoundError:
    train_df = None

def predict_survival():
    if model is None or scaler is None:
        messagebox.showerror("Error", "Model or Scaler not found. Please run AI_Project2.py first to generate 'lr_model.pkl' and 'scaler.pkl'.")
        return

    try:
        # Get inputs
        pclass = int(pclass_var.get())
        sex = 0 if sex_var.get() == "Male" else 1
        age = float(age_entry.get())
        sibsp = int(sibsp_entry.get())
        parch = int(parch_entry.get())
        fare = float(fare_entry.get())
        
        # Embarked logic
        embarked = embarked_var.get()
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0

        # Features order must match X columns: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_q, embarked_s]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Show result
        if prediction == 1:
            result_label.config(text="Prediction: Survived! \U0001F389", fg="green")
        else:
            result_label.config(text="Prediction: Did Not Survive \U0001F614", fg="red")

    except ValueError:
        messagebox.showerror("Invalid Input", "Please ensure all text fields contain valid numbers.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_survival_by_sex():
    if train_df is None:
        messagebox.showerror("Error", "Titanic_train.csv not found!")
        return
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', hue='Sex', data=train_df)
    plt.title('Survival by Sex')
    plt.show()

def show_survival_by_pclass():
    if train_df is None:
        messagebox.showerror("Error", "Titanic_train.csv not found!")
        return
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', hue='Pclass', data=train_df)
    plt.title('Survival by Passenger Class')
    plt.show()

def show_age_distribution():
    if train_df is None:
        messagebox.showerror("Error", "Titanic_train.csv not found!")
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(data=train_df, x='Age', hue='Survived', multiple='stack', kde=True)
    plt.title('Age Distribution by Survival')
    plt.show()

def show_correlation_matrix():
    if train_df is None:
        messagebox.showerror("Error", "Titanic_train.csv not found!")
        return
    plt.figure(figsize=(10, 6.5))
    numeric_train_df = train_df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

# Create the main window
root = tk.Tk()
root.title("Titanic Survival Predictor")
root.geometry("400x620")
root.resizable(False, False)

# Title
title_label = tk.Label(root, text="Titanic Survival Predictor", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Main frame for inputs
frame = tk.Frame(root)
frame.pack(pady=5)

# Pclass
tk.Label(frame, text="Passenger Class:").grid(row=0, column=0, sticky="w", pady=5)
pclass_var = tk.StringVar(value="3")
tk.OptionMenu(frame, pclass_var, "1", "2", "3").grid(row=0, column=1, pady=5, sticky="ew")

# Sex
tk.Label(frame, text="Sex:").grid(row=1, column=0, sticky="w", pady=5)
sex_var = tk.StringVar(value="Male")
tk.OptionMenu(frame, sex_var, "Male", "Female").grid(row=1, column=1, pady=5, sticky="ew")

# Age
tk.Label(frame, text="Age:").grid(row=2, column=0, sticky="w", pady=5)
age_entry = tk.Entry(frame)
age_entry.insert(0, "25")
age_entry.grid(row=2, column=1, pady=5)

# SibSp
tk.Label(frame, text="Siblings/Spouses Aboard:").grid(row=3, column=0, sticky="w", pady=5)
sibsp_entry = tk.Entry(frame)
sibsp_entry.insert(0, "0")
sibsp_entry.grid(row=3, column=1, pady=5)

# Parch
tk.Label(frame, text="Parents/Children Aboard:").grid(row=4, column=0, sticky="w", pady=5)
parch_entry = tk.Entry(frame)
parch_entry.insert(0, "0")
parch_entry.grid(row=4, column=1, pady=5)

# Fare
tk.Label(frame, text="Fare ($):").grid(row=5, column=0, sticky="w", pady=5)
fare_entry = tk.Entry(frame)
fare_entry.insert(0, "7.25")
fare_entry.grid(row=5, column=1, pady=5)

# Embarked
tk.Label(frame, text="Port of Embarkation:").grid(row=6, column=0, sticky="w", pady=5)
embarked_var = tk.StringVar(value="S")
tk.OptionMenu(frame, embarked_var, "C", "Q", "S").grid(row=6, column=1, pady=5, sticky="ew")

# Predict Button
predict_btn = tk.Button(root, text="Predict Survival", command=predict_survival, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=20)
predict_btn.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="Enter details and click Predict", font=("Helvetica", 14, "bold"))
result_label.pack(pady=5)

# Separator
tk.Frame(root, height=2, bd=1, relief="sunken").pack(fill="x", padx=20, pady=10)

# Graphs section
tk.Label(root, text="Data Visualizations", font=("Helvetica", 12, "bold")).pack(pady=5)
graph_frame = tk.Frame(root)
graph_frame.pack(pady=5)

tk.Button(graph_frame, text="Survival by Sex", command=show_survival_by_sex, width=15).grid(row=0, column=0, padx=5, pady=5)
tk.Button(graph_frame, text="Survival by Class", command=show_survival_by_pclass, width=15).grid(row=0, column=1, padx=5, pady=5)
tk.Button(graph_frame, text="Age Distribution", command=show_age_distribution, width=15).grid(row=1, column=0, padx=5, pady=5)
tk.Button(graph_frame, text="Correlation Matrix", command=show_correlation_matrix, width=15).grid(row=1, column=1, padx=5, pady=5)

if model is None:
    result_label.config(text="Models not found! Run AI_Project2.py", fg="red")
if train_df is None:
    messagebox.showwarning("Warning", "Titanic_train.csv not found! Graphs won't work.")

root.mainloop()