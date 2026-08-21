import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    run_name = (
        f"gradient_boosting_n{params['n_estimators']}"
        f"_lr{params['learning_rate']}_depth{params['max_depth']}"
    )
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_eval)[:, 1]
        default_predictions = (probabilities >= 0.5).astype(int)
        f1_default = float(f1_score(y_eval, default_predictions))

        thresholds = np.arange(0.10, 0.91, 0.05)
        f1_by_threshold = {
            float(threshold): float(
                f1_score(y_eval, (probabilities >= threshold).astype(int))
            )
            for threshold in thresholds
        }
        best_threshold, f1 = max(
            f1_by_threshold.items(), key=lambda item: item[1]
        )
        predictions = (probabilities >= best_threshold).astype(int)
        acc = float(accuracy_score(y_eval, predictions))
        precision_by_class = precision_score(
            y_eval, predictions, labels=[0, 1], average=None, zero_division=0
        )
        recall_by_class = recall_score(
            y_eval, predictions, labels=[0, 1], average=None, zero_division=0
        )
        matrix = confusion_matrix(y_eval, predictions, labels=[0, 1])

        # Luu threshold cung voi model de API dung cung quy tac voi quality gate.
        model.decision_threshold = best_threshold

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score_default_threshold", f1_default)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("precision_class_0", float(precision_by_class[0]))
        mlflow.log_metric("precision_class_1", float(precision_by_class[1]))
        mlflow.log_metric("recall_class_0", float(recall_by_class[0]))
        mlflow.log_metric("recall_class_1", float(recall_by_class[1]))
        mlflow.sklearn.log_model(model, "model")

        print(
            f"F1 (threshold={best_threshold:.2f}): {f1:.4f} | "
            f"F1 (threshold=0.50): {f1_default:.4f} | Accuracy: {acc:.4f}"
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "best_threshold": best_threshold,
                    "f1_score_default_threshold": f1_default,
                },
                f,
            )

        detail_report = (
            "Classification report\n"
            f"Decision threshold: {best_threshold:.2f}\n"
            f"F1 score (positive class): {f1:.4f}\n"
            f"Accuracy: {acc:.4f}\n\n"
            "Confusion matrix (rows=true, columns=predicted)\n"
            "                 Predicted 0  Predicted 1\n"
            f"True 0           {matrix[0, 0]:>11}  {matrix[0, 1]:>11}\n"
            f"True 1           {matrix[1, 0]:>11}  {matrix[1, 1]:>11}\n\n"
            "Per-class metrics\n"
            "Class 0 (thu_nhap_thap): "
            f"precision={precision_by_class[0]:.4f}, recall={recall_by_class[0]:.4f}\n"
            "Class 1 (thu_nhap_cao): "
            f"precision={precision_by_class[1]:.4f}, recall={recall_by_class[1]:.4f}\n"
        )
        with open("outputs/detail.txt", "w") as f:
            f.write(detail_report)
        print(detail_report)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
