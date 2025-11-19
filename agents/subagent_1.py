import pandas as pd
from langchain.tools import tool
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
)

@tool("dataset-analyser", return_direct=True)
def data_loader_analyser(filepath: str):
    """
    Load and analyse a dataset from the given CSV filepath.

    Only datasets with categorical (object/string) or numerical (int/float/nullable) columns are allowed.

    This function:
    - accepts any numeric dtype (float32/64, int32/64, nullable Int64, etc.)
    - treats object/string columns that are fully numeric as numeric
    - rejects datetime-like columns, image/video filepath columns, mixed-type object columns,
      and any other non-numeric/non-categorical dtypes
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {"status": "error", "message": f"Error reading dataset: {e}"}

    invalid_columns = []

    # common image/video extensions for filepath detection (lowercase)
    file_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".mp4", ".avi", ".mov", ".mkv", ".webm")

    for col in df.columns:
        series = df[col]
        # 1) Accept numeric dtypes (covers float32/64, int32/64, nullable dtypes, etc.)
        if is_numeric_dtype(series):
            continue

        # 2) Reject explicit datetime types
        if is_datetime64_any_dtype(series):
            invalid_columns.append((col, "datetime-like (pandas datetime dtype)"))
            continue

        # Only object/string columns remain to be checked for being categorical, numeric-strings, filepaths, etc.
        if is_object_dtype(series) or is_string_dtype(series):
            non_null = series.dropna()
            # 2a) If the object column can be fully converted to numeric -> treat as numeric (allowed)
            coerced = pd.to_numeric(non_null, errors="coerce")
            if coerced.notna().all():
                # column is numeric in content (strings of numbers) → allowed
                continue

            # 2b) Check if column is datetime-like (strings representing datetimes)
            coerced_dt = pd.to_datetime(non_null, errors="coerce", dayfirst=False)
            # consider datetime-like if a large fraction (e.g., >90%) parse successfully
            if len(non_null) > 0 and (coerced_dt.notna().sum() / len(non_null)) >= 0.90:
                invalid_columns.append((col, "datetime-like (string values parsed to datetime)"))
                continue

            # 2c) Check for image/video filepaths by extension
            # only check string endings on non-null entries
            if non_null.astype(str).str.lower().str.endswith(file_exts).any():
                invalid_columns.append((col, "image/video filepath (detected by extension)"))
                continue

            # 2d) Mixed types inside object column (e.g., ints, dicts, lists, strings mixed)
            unique_py_types = non_null.map(lambda x: type(x)).unique()
            # if there is more than one python type AND it's not just str + native numeric types cast as str, reject
            if len(unique_py_types) > 1:
                # allow case where everything is str-like (single type) — keep as categorical
                types_names = tuple(t.__name__ for t in unique_py_types)
                invalid_columns.append((col, f"mixed python types in object column: {types_names}"))
                continue

            # else: it's a pure string categorical column (allowed) — continue
            continue

        # 3) Any other dtype (bool, category, complex, sparse, object-like exotic) — decide:
        if series.dtype == "bool" or series.dtype.name == "category":
            # treat as categorical → allowed
            continue

        # Otherwise reject with dtype name
        invalid_columns.append((col, f"unsupported dtype: {series.dtype}"))

    if invalid_columns:
        return {
            "status": "rejected",
            "message": (
                "❌ Dataset rejected. Only numeric or categorical (string/object/bool/category) columns are supported. "
                "Supported datatypes are int, float, object (string), bool, and category."
            ),
            "invalid_columns": invalid_columns,
            "note": (
                "Numeric-looking strings (e.g. '3.4') are accepted (they will be interpreted as numeric). "
                "Datetime-like columns, image/video filepath columns, and mixed-type object columns are rejected."
            )
        }

    # Build final output
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # convert object columns that are fully numeric-strings to numeric explicitly for downstream use
    for col in X.select_dtypes(include=["object", "string"]).columns:
        non_null = X[col].dropna()
        coerced = pd.to_numeric(non_null, errors="coerce")
        if len(non_null) > 0 and coerced.notna().all():
            X[col] = pd.to_numeric(X[col], errors="coerce")

    categorical_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    report = {
        "shape": df.shape,
        "nulls": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    return {
        "status": "success",
        "message": "Dataset successfully analysed ✔️",
        "X": X.to_dict(orient="list"),
        "y": y.tolist(),
        "categorical_cols": categorical_cols,
        "eda_report": report,
        "target": target
    }
