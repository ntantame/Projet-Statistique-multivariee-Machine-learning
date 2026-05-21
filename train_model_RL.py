# importation des bibliotheques

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics         import (roc_auc_score, f1_score, recall_score,
                                     classification_report)

# 1. CHARGEMENT 
CSV = "CDIR81FL.csv"
print(f"[1/7] Chargement de '{CSV}'...")
data = pd.read_csv(CSV, sep=";", low_memory=False)
print(f"      {data.shape[0]} lignes, {data.shape[1]} colonnes")

# 2. CATÉGORISATION PARITÉ & CPN 
print("[2/7] Catégorisation parité & CPN...")

def cat_parite(x):
    if pd.isna(x) or x <= 2: return "1-2"
    elif x <= 4:              return "3-4"
    else:                     return "5+"

def cat_cpn(x):
    if pd.isna(x) or x == 0: return "0"
    elif x <= 3:              return "1-3"
    else:                     return "4+"

data["parite"]  = pd.Categorical(data["v201"].apply(cat_parite),
                                  categories=["1-2","3-4","5+"])
data["cpn_cat"] = pd.Categorical(data["cpn_visites"].apply(cat_cpn),
                                  categories=["0","1-3","4+"])
data.drop(columns=["v201","cpn_visites"], inplace=True, errors="ignore")

# 3. TYPAGE CATÉGORIEL 
vars_cat = ["v013","v024","v025","v106","v130","v190","v717","v701",
            "v157","v158","v159","v743a","v743b","v743d","parite","cpn_cat"]
for c in vars_cat:
    if c in data.columns:
        data[c] = data[c].astype("category")

# 4. ENCODAGE ONE-HOT 
print("[3/7] Encodage one-hot...")
data_dum = pd.get_dummies(data, columns=vars_cat, drop_first=True)
bool_c   = data_dum.select_dtypes("bool").columns
data_dum[bool_c] = data_dum[bool_c].astype(int)
print(f"      {data_dum.shape[1]} colonnes après encodage")

# 5. SPLIT TRAIN / TEST 
print("[4/7] Split 80/20 stratifié...")
exclure  = ["caseid","v005","v021","SBA","poids_final"]
features = [c for c in data_dum.columns if c not in exclure]
X = data_dum[features]
y = data_dum["SBA"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"      Train={X_train.shape}, Test={X_test.shape}")
print(f"      SBA=1 train: {y_train.mean():.1%}  |  SBA=1 test: {y_test.mean():.1%}")

# 6. PAS DE STANDARDISATION (variables catégorielles one-hot)
print("[5/7] Pas de standardisation (variables catégorielles)...")

# 7. ENTRAÎNEMENT : class_weight='balanced' 
print("[6/7] GridSearchCV — Régression Logistique (class_weight=balanced)...")
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"   
    ),
    param_grid={"C": [0.001, 0.01, 0.1]},
    cv=cv_inner,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print(f"      Meilleur C  : {grid.best_params_['C']}")
print(f"      AUC CV      : {grid.best_score_:.4f}")
print(f"      Intercept   : {best_model.intercept_[0]:.4f}")

# 8. RECHERCHE DU SEUIL OPTIMAL 
print("[7/7] Recherche du seuil de décision optimal...")
probas_train = best_model.predict_proba(X_train)[:, 1]
probas_test  = best_model.predict_proba(X_test)[:,  1]

best_threshold = 0.50  # Seuil fixé à 0.50
best_f1_0      = 0.0
threshold_log  = []

for t in np.arange(0.20, 0.80, 0.01):
    preds  = (probas_train >= t).astype(int)
    f1_0   = f1_score(y_train, preds, pos_label=0, zero_division=0)
    rec_1  = recall_score(y_train, preds, pos_label=1, zero_division=0)
    rec_0  = recall_score(y_train, preds, pos_label=0, zero_division=0)
    threshold_log.append((round(t,2), round(f1_0,3), round(rec_1,3), round(rec_0,3)))
    if f1_0 > best_f1_0 and rec_1 >= 0.75:
        best_f1_0 = f1_0

print(f"      Seuil fixé    : {best_threshold}  "
      f"(F1-SBA=0={best_f1_0:.3f})")

# 9. RAPPORT FINAL 
auc_test  = roc_auc_score(y_test, probas_test)
preds_opt = (probas_test >= best_threshold).astype(int)
preds_def = (probas_test >= 0.50).astype(int)

print(f"\n{'='*55}")
print(f"  AUC Test : {auc_test:.4f}")
print(f"{'='*55}")
print(f"\n Seuil par défaut (0.50)")
print(classification_report(y_test, preds_def,
      target_names=["Non-Assistée (0)","Assistée (1)"], digits=3))
print(f" Seuil optimal ({best_threshold}) ")
print(classification_report(y_test, preds_opt,
      target_names=["Non-Assistée (0)","Assistée (1)"], digits=3))

# 10. SAUVEGARDE 
os.makedirs("model_files", exist_ok=True)
joblib.dump(best_model,            "model_files/model.pkl")
joblib.dump(list(X_train.columns), "model_files/columns.pkl")
joblib.dump(best_threshold,        "model_files/threshold.pkl")

print(f"\n{'='*55}")
print("  Fichiers sauvegardés dans model_files/")
print(f"    model.pkl       => Régression Logistique (balanced, C={grid.best_params_['C']})")
print(f"    columns.pkl     => {len(features)} colonnes prédicteurs")
print(f"    threshold.pkl   => seuil optimal = {best_threshold}")
print(f"\n  Lancement de l'application streamlit")