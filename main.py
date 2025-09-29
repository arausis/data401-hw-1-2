import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression 

if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("mahmoudelhemaly/students-grading-dataset")
    df = pd.read_csv(f"{path}/Students Performance Dataset.csv")
    print('What is the average number of hours students study per week?')
    print(np.mean(df["Study_Hours_per_Week"]))
    print()

    print('Given a student, what is the average number of hours of sleep they receive per night?')
    print(np.mean(df["Sleep_Hours_per_Night"]))
    print()

    print('What proportion of students come from a home with no internet access?')
    df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].map(lambda x : x == "No")
    print(f'{np.mean(df["Internet_Access_at_Home"]) * 100}% of students come from a home with no internet access')
    print()

    print("How does a student’s participation score differ between income levels (low, medium, high)?")
    print(df.groupby("Family_Income_Level")["Total_Score"].mean())
    print("No, not significantly")
    print()

    print("Does parent education level impact students' average score on quizzes and assignments?")
    print(df.groupby("Parent_Education_Level")[['Quizzes_Avg', 'Assignments_Avg']].mean())
    print("No, not significantly")
    print()

    print("Does the student's stress level have a huge impact on how well they perform on final exams?")
    x_vals = df["Stress_Level (1-10)"]
    y_vals = df["Final_Score"]
    r = np.corrcoef(x_vals, y_vals)[0, 1]
    print(f"No, the correlation coefficient is only {r} which is not significant")
    print()

    print("Does participation in extracurricular activities impact the amount of hours that a student studies per week?")
    print(df.groupby('Extracurricular_Activities')["Study_Hours_per_Week"].mean())
    print("No, not significantly")

