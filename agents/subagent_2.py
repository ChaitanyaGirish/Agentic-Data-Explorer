import pandas as pd
from langchain.tools import tool
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

@tool("model-selector", return_direct=True)
def model_selector_tool(filepath: str, target: str):
    """
    Load a CSV, split into X (features) and y (target),
    preprocess numerical and categorical columns,
    and evaluate several standard classifiers using 5-fold CV.

    Returns a dict with mean and std accuracy for each model.
    """
    # 1) Load data
    df = pd.read_csv(filepath)
    y = df[target]
    X = df.drop(target, axis=1)
    print("Loading data done successfully.")

    # 2) Identify feature types
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    print("step 2 successful.")

    # 3) Define preprocessing
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 4) Define candidate models
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
        "svc": SVC(),  # default RBF kernel
    }

    # 5) Evaluate each model with a pipeline
    results = {}
    for name, clf in models.items():
        clf_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ])

        scores = cross_val_score(clf_pipeline, X, y, cv=5, scoring="accuracy")

        results[name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        }

    return results
