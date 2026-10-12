import argparse

import kagglehub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Target_Dropout_Next_Sem"
FEATURES = [
    "Family_Income", "Work_Hours", "Emergency_Expense", "Sem_GPA", "Attendance",
    "LMS_Logins", "Financial_Stress", "Age", "Household_Size", "Semester",
    "Course_Load", "Tuition_Base", "Advising_Visits", "Failed_Courses",
    "First_Generation", "Scholarship", "Gender", "Housing_Status",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--C", type=float, default=1.0, help="inverse regularization strength (smaller = stronger)")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment-name", default="student-dropout-logreg")
    return p.parse_args()


def load_data():
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "razanihababdellatif/student-retention-and-academic-performance-data",
        "academic_survival_longitudinal.csv",
    )


def clean(df):
    df["Family_Income"] = df["Family_Income"].fillna(df["Family_Income"].median())
    df["LMS_Logins"] = df["LMS_Logins"].fillna(df["LMS_Logins"].median())
    df["Attendance"] = df["Attendance"].clip(0, 100)
    df["Gender"] = df["Gender"].replace(
        {"female": "Female", "F": "Female", "male": "Male", "M": "Male"}
    )
    return df


def split(df, seed):
    X = pd.get_dummies(df[FEATURES], drop_first=True)
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def train(X_train, y_train, args):
    class_weight = None if args.class_weight == "none" else args.class_weight
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, class_weight=class_weight,
                           max_iter=args.max_iter, random_state=args.seed),
    )
    return model.fit(X_train, y_train)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "dropout_recall": recall_score(y_test, y_pred),
        "dropout_precision": precision_score(y_test, y_pred, zero_division=0),
        "dropout_f1": f1_score(y_test, y_pred, zero_division=0),
    }
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig("confusion_matrix.png")
    plt.close()
    return metrics


def main():
    # read arguments
    args = parse_args()
    mlflow.set_experiment(args.experiment_name)

    # load and split data
    df = clean(load_data())
    X_train, X_test, y_train, y_test = split(df, args.seed)

    # start mlflow run
    with mlflow.start_run():
        # log all the parameters
        for name, value in vars(args).items():
            mlflow.log_param(name, value)

        # train the model
        model = train(X_train, y_train, args)

        # evaluate the model
        metrics = evaluate(model, X_test, y_test)

        # log metrics, artifacts and model.
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_artifact("confusion_matrix.png")
        mlflow.sklearn.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
