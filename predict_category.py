"""
predict_category.py
Incarca modelul .pkl si ofera o interfata interactiva pentru a prezice categoria unui produs dupa titlu.

Rulare:
    python predict_category.py --model models/category_model.pkl
"""

import argparse
import joblib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/category_model.pkl", help="Calea catre modelul salvat (.pkl)")
    args = ap.parse_args()

    clf = joblib.load(args.model)
    print(" Model incarcat. Scrie un titlu (tasteaza 'exit' pentru a iesi).")

    while True:
        title = input("\nTitlu produs: ").strip()
        if title.lower() in {"exit", "quit", "q"}:
            print("La revedere!")
            break

        pred = clf.predict([title])[0]
        print(f" Categoria prezisa: {pred}")

        # scoruri pentru top-5, daca modelul le ofera
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function([title])[0]
            classes = clf.classes_
            idx = np.argsort(scores)[::-1][:5]
            print("Top 5 categorii probabile:")
            for i in idx:
                print(f"  - {classes[i]} (score: {scores[i]:.3f})")
        elif hasattr(clf, "predict_proba"):
            probs = clf.predict_proba([title])[0]
            classes = clf.classes_
            idx = np.argsort(probs)[::-1][:5]
            print("Top 5 categorii probabile:")
            for i in idx:
                print(f"  - {classes[i]} ({probs[i]:.2%})")

if __name__ == "__main__":
    main()
