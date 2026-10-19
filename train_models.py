import time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,roc_curve)

warnings.filterwarnings("ignore")
RS = 42

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns="cardio"), train["cardio"]
X_val, y_val = val.drop(columns="cardio"), val["cardio"]
X_test, y_test = test.drop(columns="cardio"), test["cardio"]

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),("m", LogisticRegression(max_iter=1000, random_state=RS))]),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=25))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=RS),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3,learning_rate=0.1, random_state=RS),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,random_state=RS),
    "Neural Network (MLP)": Pipeline([("sc", StandardScaler()),("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,early_stopping=True, random_state=RS))]),
}

# # Optional extra "state-of-the-art" models (only used if the libraries are installed)
# try:
#     from xgboost import XGBClassifier
#     models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
#                                       eval_metric="logloss", random_state=RS, n_jobs=-1)
# except ImportError:
#     print("xgboost not installed - skipping (pip install xgboost to include it)")
# try:
#     from lightgbm import LGBMClassifier
#     models["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RS,
#                                         verbose=-1)
# except ImportError:
#     print("lightgbm not installed - skipping (pip install lightgbm to include it)")


def scores(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred),
                Recall=recall_score(y, pred), F1=f1_score(y, pred),
                ROC_AUC=roc_auc_score(y, proba))

rows, fitted = [], {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    secs = time.time() - t0
    v, t = scores(model, X_val, y_val), scores(model, X_test, y_test)
    rows.append({"Model": name,
                 "Val_Accuracy": v["Accuracy"], "Val_ROC_AUC": v["ROC_AUC"],
                 "Test_Accuracy": t["Accuracy"], "Test_Precision": t["Precision"],
                 "Test_Recall": t["Recall"], "Test_F1": t["F1"], "Test_ROC_AUC": t["ROC_AUC"],
                 "Train_Time_s": secs})
    fitted[name] = model
    print(f"{name:22s} val acc={v['Accuracy']:.4f}  test acc={t['Accuracy']:.4f}  "
          f"AUC={t['ROC_AUC']:.4f}  ({secs:.1f}s)")

results = pd.DataFrame(rows).sort_values("Val_ROC_AUC", ascending=False).reset_index(drop=True)
results.round(4).to_csv("outputs/model_comparison.csv", index=False)
print("\n", results.round(4).to_string(index=False))

best_name = results.loc[0, "Model"]
best = fitted[best_name]
joblib.dump({"model": best, "features": list(X_train.columns), "name": best_name},"models/best_model.joblib")
print("\nBest model:", best_name)


plot_df = results.set_index("Model")[["Test_Accuracy", "Test_F1", "Test_ROC_AUC"]]
ax = plot_df.plot(kind="bar", figsize=(10, 5), ylim=(0.6, 0.85))
ax.set_title("Model comparison on the test set"); ax.set_ylabel("Score")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(7, 6))
for name, m in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test set)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150); plt.close()

ConfusionMatrixDisplay(confusion_matrix(y_test, best.predict(X_test)),
                       display_labels=["No disease", "Disease"]).plot(cmap="Blues")
plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

from sklearn.inspection import permutation_importance
pi = permutation_importance(best, X_val, y_val, n_repeats=5, random_state=RS,
                            scoring="roc_auc", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X_val.columns).sort_values()
imp.plot(kind="barh", figsize=(7, 5), title=f"Permutation importance - {best_name}")
plt.xlabel("Drop in ROC-AUC when feature is shuffled"); plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()
imp.sort_values(ascending=False).round(4).to_csv("outputs/feature_importance.csv", header=["importance"])

clean = pd.read_csv("data/cardio_clean.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
clean.groupby("cardio")["age_years"].plot(kind="hist", alpha=0.5, bins=25, ax=axes[0], legend=True)
axes[0].set_title("Age by disease (0/1)")
clean.boxplot(column="ap_hi", by="cardio", ax=axes[1]); axes[1].set_title("Systolic BP by disease")
clean.boxplot(column="bmi", by="cardio", ax=axes[2]); axes[2].set_title("BMI by disease")
plt.suptitle(""); plt.tight_layout(); plt.savefig("outputs/eda.png", dpi=150); plt.close()

import seaborn as sns
plt.figure(figsize=(9, 7))
sns.heatmap(clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation matrix"); plt.tight_layout()
plt.savefig("outputs/correlation.png", dpi=150); plt.close()
print("Plots saved in outputs/")
