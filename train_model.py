"""
train_model.py
Antreneaza un model care sugereaza categoria pe baza coloanei "Product Title"
si salveaza un pipeline sklearn (TF-IDF + LinearSVC) în .pkl

Rulare:
    python train_model.py --data products.csv --model_out models/category_model.pkl
"""

import argparse
from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_columns(df: pd.DataFrame):
    lookup = {c.strip().lower(): c for c in df.columns}
    title_col = lookup.get("product title") or lookup.get("title")
    cat_col   = lookup.get("category label") or lookup.get("category") or lookup.get("label")
    if not title_col or not cat_col:
        raise ValueError(
            f"Nu am gasit coloanele necesare. Coloane disponibile: {list(df.columns)}.\n"
            "Caut 'Product Title' si 'Category Label' (insensibil la majuscule)."
        )
    return title_col, cat_col

def clean_title(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Calea către products.csv")
    ap.add_argument("--model_out", default="models/category_model.pkl", help="Unde salvăm modelul .pkl")
    ap.add_argument("--limit", type=int, default=None, help="Antrenează pe primele N rânduri (debug rapid)")
    args = ap.parse_args()

    # 1) Incarcare
    df = pd.read_csv(Path(args.data))
    df = clean_columns(df)
    title_col, cat_col = infer_columns(df)

    # 2) Curatare de baza
    df = df.dropna(subset=[title_col, cat_col]).copy()
    df[cat_col]   = df[cat_col].astype(str).str.strip()
    df[title_col] = df[title_col].astype(str).map(clean_title)

    if args.limit:
        df = df.head(args.limit).copy()

    # 3) Split
    X = df[title_col].values
    y = df[cat_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Pipeline (TF-IDF + LinearSVC)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            lowercase=True,
            sublinear_tf=True,
            max_features=120_000
        )),
        ("clf", LinearSVC())
    ])

    # 5) Antrenare + Evaluare
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification report (trunchiat la 3000 caractere):\n")
    rep = classification_report(y_test, y_pred, zero_division=0)
    print(rep[:3000])

    # 6) salvare model
    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"\nModel salvat la: {out.resolve()}")

if __name__ == "__main__":
    main()
