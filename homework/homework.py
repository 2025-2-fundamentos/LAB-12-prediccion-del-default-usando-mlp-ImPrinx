# flake8: noqa: E501
import numpy as np
import pandas as pd
import json
import os
import gzip
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent  # Carpeta LAB-12...

class DataPreprocessor:
    @staticmethod
    def read_dataset(relative_path: str) -> pd.DataFrame:
        return pd.read_csv(BASE_DIR / relative_path, compression="zip")

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        processed = processed[
            (processed["MARRIAGE"] != 0) & (processed["EDUCATION"] != 0)
        ]
        processed.loc[processed["EDUCATION"] >= 4, "EDUCATION"] = 4

        processed = (
            processed.rename(columns={"default payment next month": "default"})
            .drop("ID", axis=1)
            .dropna()
        )

        return processed


class ModelBuilder:
    def __init__(self):
        self.categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
        self.target_col = "default"

    def create_feature_pipeline(self, X):
        numeric_cols = [col for col in X.columns if col not in self.categorical_cols]

        preprocessor = ColumnTransformer(
            [
                ("categorical", OneHotEncoder(), self.categorical_cols),
                ("numeric", StandardScaler(), numeric_cols),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("feature_selection", SelectKBest(score_func=f_classif)),
                ("pca", PCA()),
                ("classifier", MLPClassifier(max_iter=15000, random_state=21)),
            ]
        )

        param_grid = {
            "pca__n_components": [None],
            "feature_selection__k": [20],
            "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
            "classifier__alpha": [0.26],
            "classifier__learning_rate_init": [0.001],
        }

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=10,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
        )

    def train_model(self, X, y):
        grid_search = self.create_feature_pipeline(X)
        return grid_search.fit(X, y)


class MetricsCalculator:
    @staticmethod
    def compute_performance_metrics(dataset_name: str, y_true, y_pred) -> dict:
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

    @staticmethod
    def compute_confusion_matrix(dataset_name: str, y_true, y_pred) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }


class CreditDefaultPredictor:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model_builder = ModelBuilder()
        self.metrics_calculator = MetricsCalculator()

    def prepare_data(self, df: pd.DataFrame):
        X = df.drop("default", axis=1)
        y = df["default"]
        return X, y

    def save_model(self, model, relative_path: str):
        path = BASE_DIR / relative_path
        os.makedirs(path.parent, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(model, f)

    def save_metrics(self, metrics_list: list, relative_path: str):
        path = BASE_DIR / relative_path
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for metric in metrics_list:
                f.write(json.dumps(metric) + "\n")

    def run_pipeline(self):
        train_df = self.data_preprocessor.read_dataset("files/input/train_data.csv.zip")
        test_df = self.data_preprocessor.read_dataset("files/input/test_data.csv.zip")

        train_clean = self.data_preprocessor.preprocess_data(train_df)
        test_clean = self.data_preprocessor.preprocess_data(test_df)

        X_train, y_train = self.prepare_data(train_clean)
        X_test, y_test = self.prepare_data(test_clean)

        model = self.model_builder.train_model(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        metrics = [
            self.metrics_calculator.compute_performance_metrics("train", y_train, train_preds),
            self.metrics_calculator.compute_performance_metrics("test", y_test, test_preds),
            self.metrics_calculator.compute_confusion_matrix("train", y_train, train_preds),
            self.metrics_calculator.compute_confusion_matrix("test", y_test, test_preds),
        ]

        self.save_model(model, "files/models/model.pkl.gz")
        self.save_metrics(metrics, "files/output/metrics.json")


if __name__ == "__main__":
    predictor = CreditDefaultPredictor()
    predictor.run_pipeline()