# train_model_RF.py - Version corrigée avec encodage personnalisé identique au notebook
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report

print(" Entraînement Random Forest - EDS Cameroun (encodage personnalisé)")

# ====================== 1. CHARGEMENT ======================
CSV = "PrCMIR71FL.csv"
print(f"[1/7] Chargement de '{CSV}'...")
data = pd.read_csv(CSV, sep=";", low_memory=False)
print(f"      {data.shape[0]} lignes, {data.shape[1]} colonnes")

# ====================== 2. CRÉATION de parite et cpn_cat (identique au notebook) ======================
def cat_parite(x):
    if pd.isna(x) or x <= 2:
        return "1"
    elif x <= 4:
        return "2"
    else:
        return "3"

def cat_cpn(x):
    if pd.isna(x) or x == 0:
        return "0"
    else:
        return "1"

data["parite"] = data["parite"].apply(cat_parite)
data["cpn_cat"] = data["ANC"].apply(cat_cpn)

print(" parite et cpn_cat créées")

# ====================== 3. DÉFINITION DE L'ORDRE PERSONNALISÉ (identique au notebook) ======================
custom_order = {
    'V013': [1, 2, 3, 4, 5, 6, 7],
    'V024': [5, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],  # 5 = Extrême-Nord en référence
    'V025': [2, 1],                                    # 2 = Rural en référence
    'V106': [0, 1, 2, 3],                              # 0 = Aucun niveau en référence
    'V130': [4, 1, 2, 3, 5, 7, 96],                    # 4 = Musulman en référence
    'V190': [1, 2, 3, 4, 5],                           # 1 = Plus pauvre en référence
    'V701': [0, 1, 2, 3, 8],                           # 0 = Aucun niveau en référence
    'Media_exposure': [0, 1],
    'autonomie_decisionnelle': [1, 2, 3],
    'attitude_violence': [1, 0],                       # 1 = Rejette la violence en référence
    'parite': ['3', '2', '1'],
    'cpn_cat': ['0', '1']
}

# Liste des variables catégorielles à transformer
cat_vars = list(custom_order.keys())

# ====================== 4. APPLICATION DE L'ORDRE PERSONNALISÉ ======================
print("[4/7] Application de l'ordre personnalisé sur les variables catégorielles...")
for var in cat_vars:
    if var in data.columns:
        data[var] = pd.Categorical(data[var], categories=custom_order[var], ordered=True)
    else:
        print(f"AVERTISSEMENT : la colonne '{var}' n'existe pas dans les données")

# ====================== 5. ONE-HOT ENCODING AVEC drop_first=True ======================
print("[5/7] Encodage one-hot avec drop_first=True...")
data_dum = pd.get_dummies(data, columns=cat_vars, drop_first=True)

# Conversion des booléens en entiers (comme dans le notebook)
bool_cols = data_dum.select_dtypes(include='bool').columns
data_dum[bool_cols] = data_dum[bool_cols].astype(int)

# Vérification des colonnes générées (optionnel)
print("Exemples de colonnes one-hot créées :")
for var in cat_vars:
    cols = [c for c in data_dum.columns if c.startswith(f"{var}_")]
    if cols:
        print(f"  {var}: {cols[:3]}...")

# ====================== 6. SÉPARATION DES FEATURES ET DE LA CIBLE ======================
exclure = ["CASEID", "V005", "V021", "V023", "SBA", "poids_final","ANC"]
features = [c for c in data_dum.columns if c not in exclure]

X = data_dum[features]
y = data_dum["SBA"]

print(f"Nombre de features finales : {len(features)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ====================== 7. RANDOM FOREST AVEC GRID SEARCH ======================
print("[6/7] Entraînement Random Forest avec GridSearchCV...")

param_grid = {
    'n_estimators': [100, 200, 300],           
    'max_depth': [10, 15, 20, None],            
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],             
    'max_features': ['sqrt', 'log2', None],     
    'class_weight': ['balanced', 'balanced_subsample']
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print(f" Meilleurs paramètres : {grid.best_params_}")
print(f" AUC moyenne (CV) : {grid.best_score_:.4f}")

# ====================== 8. OPTIMISATION DU SEUIL ======================
print("[7/7] Optimisation du seuil pour la classe 'Non assistée'...")
probas_train = best_model.predict_proba(X_train)[:, 1]
probas_test = best_model.predict_proba(X_test)[:, 1]

best_threshold = 0.50
best_f1_0 = 0.0

for t in np.arange(0.20, 0.80, 0.01):
    preds = (probas_train >= t).astype(int)
    f1_0 = f1_score(y_train, preds, pos_label=0, zero_division=0)
    if f1_0 > best_f1_0:
        best_f1_0 = f1_0
        best_threshold = round(t, 3)

print(f" Meilleur seuil : {best_threshold}")

# Évaluation finale sur le test
auc_test = roc_auc_score(y_test, probas_test)
preds_test = (probas_test >= best_threshold).astype(int)

print(f"\n{'='*70}")
print(f" AUC sur le test : {auc_test:.4f}")
print(f"{'='*70}")
print(classification_report(y_test, preds_test,
      target_names=["Non-Assistée (0)", "Assistée (1)"], digits=3))

# ====================== 9. SAUVEGARDE ======================
os.makedirs("model_files", exist_ok=True)
joblib.dump(best_model, "model_files/model.pkl")
joblib.dump(list(X_train.columns), "model_files/columns.pkl")
joblib.dump(best_threshold, "model_files/threshold.pkl")

print("\n Entraînement terminé avec succès !")
print(" Fichiers sauvegardés dans le dossier 'model_files/'.")