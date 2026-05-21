"""
app.py — Application Streamlit : Prédiction de l'accouchement assisté (SBA) en RDC
Auteur  : NTANTAME DIHEWOU TED HARRIS — Master 1 Data Science, IUSJ
Données : EDS-RDC 2023-2024 (CDIR81FL)
Modèle  : Régression Logistique (class_weight=balanced) | AUC ≈ 0.83
Seuil   : Optimal (fixé à 0.50, chargé depuis threshold.pkl)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
from datetime import datetime


#  CONSTANTES

HISTORY_FILE = "sba_history.json" 


#  CONFIG PAGE

st.set_page_config(
    page_title="SBA Prédicteur RDC",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded",
)


#  CSS GLOBAL

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0A0F1E; color: #E2E8F0; }
.stApp { background-color: #0A0F1E; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0D1B2A,#0A1628); border-right:1px solid #1E3A5F; }
[data-testid="stSidebar"] * { color:#CBD5E1 !important; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:#F8FAFC !important; }
.stButton>button { background:linear-gradient(135deg,#0EA5E9,#6366F1) !important; color:#fff !important; border:none !important; border-radius:12px !important; padding:.65rem 1.6rem !important; font-weight:700 !important; font-size:.92rem !important; box-shadow:0 4px 20px rgba(14,165,233,.35) !important; transition:all .2s !important; }
.stButton>button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(14,165,233,.45) !important; }
.stSelectbox label { color:#94A3B8 !important; font-size:.8rem !important; font-weight:600 !important; letter-spacing:.4px !important; text-transform:uppercase !important; }
.stSelectbox>div>div { background:#0F1E35 !important; border:1.5px solid #1E3A5F !important; border-radius:10px !important; color:#E2E8F0 !important; font-size:.9rem !important; }
.stSelectbox>div>div:focus-within { border-color:#38BDF8 !important; }
[data-testid="stForm"] { background:#0D1B2A; border:1px solid #1E3A5F; border-radius:16px; padding:1.5rem 2rem; }
[data-testid="stMetric"] { background:#0D1B2A; border:1px solid #1E3A5F; border-radius:12px; padding:1rem 1.2rem; }
[data-testid="stMetricLabel"] { color:#64748B !important; font-size:.75rem !important; font-weight:700 !important; text-transform:uppercase !important; letter-spacing:.5px !important; }
[data-testid="stMetricValue"] { color:#38BDF8 !important; font-family:'Space Grotesk',sans-serif !important; font-size:1.6rem !important; }
.stProgress>div>div { background:#1E3A5F !important; border-radius:99px !important; }
.stProgress>div>div>div { background:linear-gradient(90deg,#0EA5E9,#6366F1) !important; border-radius:99px !important; }
hr { border-color:#1E3A5F !important; margin:1.5rem 0 !important; }
.streamlit-expanderHeader { background:#0D1B2A !important; border-radius:10px !important; }
[data-testid="stDataFrame"] { border-radius:12px !important; overflow:hidden !important; }
.hero-card { background:linear-gradient(135deg,#0D1B2A,#0F2040,#0D1B2A); border:1px solid #1E3A5F; border-radius:20px; padding:2.5rem; text-align:center; margin-bottom:2rem; }
.stat-card { background:#0D1B2A; border:1px solid #1E3A5F; border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem; text-align:center; }
.step-card { background:#0D1B2A; border:1px solid #1E3A5F; border-radius:14px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.result-main-yes { background:linear-gradient(135deg,#052E16,#065F46); border:2px solid rgba(52,211,153,.4); border-radius:20px; padding:2rem; text-align:center; margin-bottom:1.5rem; }
.result-main-no  { background:linear-gradient(135deg,#2D0A0A,#7F1D1D); border:2px solid rgba(248,113,113,.4); border-radius:20px; padding:2rem; text-align:center; margin-bottom:1.5rem; }
.result-main-warn { background:linear-gradient(135deg,#1C1200,#78350F); border:2px solid rgba(251,191,36,.4); border-radius:20px; padding:2rem; text-align:center; margin-bottom:1.5rem; }
.risk-box-rouge  { background:#2D0A0A; border:1.5px solid #F87171; border-radius:12px; padding:1rem 1.2rem; }
.risk-box-orange { background:#1C1200; border:1.5px solid #FBBF24; border-radius:12px; padding:1rem 1.2rem; }
.risk-box-vert   { background:#052E16; border:1.5px solid #34D399; border-radius:12px; padding:1rem 1.2rem; }
.info-pill { display:inline-block; background:#0EA5E915; color:#38BDF8; border:1px solid #0EA5E940; border-radius:20px; padding:4px 14px; font-size:.78rem; font-weight:600; margin:3px; }
</style>
""", unsafe_allow_html=True)



#  CHARGEMENT DU MODÈLE + SEUIL

@st.cache_resource(show_spinner=False)
def load_model():
    base = "model_files"
    try:
        m   = joblib.load(os.path.join(base, "model.pkl"))
        c   = joblib.load(os.path.join(base, "columns.pkl"))
        # Seuil optimal sauvegardé par train_model.py
        thr = joblib.load(os.path.join(base, "threshold.pkl"))
        return m, c, thr, True
    except Exception as e:
        return None, None, 0.5, False

model, MODEL_COLS, THRESHOLD, MODEL_OK = load_model()



#  MAPPINGS LISIBLES → CODES EDS

MAP_AGE       = {"15–19 ans":1,"20–24 ans":2,"25–29 ans":3,"30–34 ans":4,
                 "35–39 ans":5,"40–44 ans":6,"45–49 ans":7}
MAP_RESIDENCE = {"Milieu urbain":1,"Milieu rural":2}
MAP_REGION    = {"Kinshasa":1,"Kongo Central":2,"Kwango":3,"Kwilu":4,
                 "Mai-Ndombe":5,"Équateur":6,"Nord Ubangi":7,"Sud Ubangi":8,
                 "Mongala":9,"Tshuapa":10,"Tshopo":11,"Bas-Uele":12,
                 "Haut Uele":13,"Ituri":14,"Nord-Kivu":15,"Sud-Kivu":16,
                 "Maniema":17,"Haut-Katanga":18,"Haut Lomami":19,"Lualaba":20,
                 "Tanganyika":21,"Lomami":22,"Sankuru":23,"Kasaï Oriental":24,
                 "Kasaï":25,"Kasaï Central":26}
MAP_EDUC      = {"Aucune instruction":0,"Niveau primaire":1,
                 "Niveau secondaire":2,"Niveau supérieur (université)":3}
MAP_EDUC_MARI = {"Aucune instruction":0,"Niveau primaire":1,
                 "Niveau secondaire":2,"Niveau supérieur (université)":3,"Ne sait pas":8}
MAP_RICHESSE  = {"Très pauvre":1,"Pauvre":2,"Revenu moyen":3,"Aisé":4,"Très aisé":5}
MAP_OCCUP     = {"Sans emploi":0,"Agriculture (à son compte)":6,
                 "Services / Commerce":4,"Personnel professionnel ou cadre":1,
                 "Travail manuel qualifié":7,"Travail manuel non qualifié":8,
                 "Employée de maison":5,"Secrétariat / Bureau":3,"Ne sait pas":98}
MAP_CPN       = {"Aucune visite":"0","1 à 3 visites":"1-3",
                 "4 visites ou plus (recommandé par l'OMS)":"4+"}
MAP_PARITE    = {"1 à 2 enfants":"1-2","3 à 4 enfants":"3-4","5 enfants ou plus":"5+"}
MAP_FREQ      = {"Jamais":0,"Moins d'une fois par semaine":1,"Au moins une fois par semaine":2}
MAP_DEC       = {"La femme seule":1,"La femme et son mari ensemble":2,
                 "Le mari seul":3,"Une autre personne":6}
MAP_RELIGION  = {"Catholique":1,"Protestant":2,
                 "Charismatique / Apostolique / Non-dénominationnel":96,
                 "Kimbanguiste":5,"Musulman":3,"Témoin de Jéhovah":4,
                 "Animiste / Religion traditionnelle":7,
                 "Armée du Salut":6,"Aucune religion":8,"Autre":9}

OPTS_FREQ    = list(MAP_FREQ.keys())
OPTS_DEC     = list(MAP_DEC.keys())
OPTS_REGIONS = list(MAP_REGION.keys())



#  CONSTRUCTION DU VECTEUR D'ENTRÉE

def build_vector(fd: dict, columns: list) -> pd.DataFrame:
    """
    FIX : on définit les catégories COMPLÈTES pour chaque variable avant d'encoder,
    forçant pd.get_dummies à générer TOUTES les colonnes attendues même avec 1 ligne.
    """
    raw = {
        "v013":    MAP_AGE[fd["age"]],
        "v024":    MAP_REGION[fd["region"]],
        "v025":    MAP_RESIDENCE[fd["residence"]],
        "v106":    MAP_EDUC[fd["education"]],
        "v701":    MAP_EDUC_MARI[fd["edu_mari"]],
        "v130":    MAP_RELIGION[fd["religion"]],
        "v190":    MAP_RICHESSE[fd["richesse"]],
        "v717":    MAP_OCCUP[fd["occupation"]],
        "v157":    MAP_FREQ[fd["journal"]],
        "v158":    MAP_FREQ[fd["radio"]],
        "v159":    MAP_FREQ[fd["tv"]],
        "v743a":   MAP_DEC[fd["dec_sante"]],
        "v743b":   MAP_DEC[fd["dec_achat"]],
        "v743d":   MAP_DEC[fd["dec_visite"]],
        "cpn_cat": MAP_CPN[fd["cpn"]],
        "parite":  MAP_PARITE[fd["parite"]],
    }
    df = pd.DataFrame([raw])

    df["v013"]  = pd.Categorical(df["v013"],  categories=[1,2,3,4,5,6,7])
    df["v024"]  = pd.Categorical(df["v024"],  categories=list(range(1,27)))
    df["v025"]  = pd.Categorical(df["v025"],  categories=[1,2])
    df["v106"]  = pd.Categorical(df["v106"],  categories=[0,1,2,3])
    df["v701"]  = pd.Categorical(df["v701"],  categories=[0,1,2,3,8])
    df["v130"]  = pd.Categorical(df["v130"],  categories=[1,2,3,4,5,6,7,8,9,96])
    df["v190"]  = pd.Categorical(df["v190"],  categories=[1,2,3,4,5])
    df["v717"]  = pd.Categorical(df["v717"],  categories=[0,1,3,4,5,6,7,8,98])
    df["v157"]  = pd.Categorical(df["v157"],  categories=[0,1,2])
    df["v158"]  = pd.Categorical(df["v158"],  categories=[0,1,2])
    df["v159"]  = pd.Categorical(df["v159"],  categories=[0,1,2])
    df["v743a"] = pd.Categorical(df["v743a"], categories=[1,2,3,6])
    df["v743b"] = pd.Categorical(df["v743b"], categories=[1,2,3,6])
    df["v743d"] = pd.Categorical(df["v743d"], categories=[1,2,3,6])
    df["cpn_cat"] = pd.Categorical(df["cpn_cat"], categories=["0","1-3","4+"])
    df["parite"]  = pd.Categorical(df["parite"],  categories=["1-2","3-4","5+"])

    vars_cat = ["v013","v024","v025","v106","v130","v190","v717","v701",
                "v157","v158","v159","v743a","v743b","v743d","parite","cpn_cat"]
    df_enc = pd.get_dummies(df, columns=vars_cat, drop_first=True)
    bc = df_enc.select_dtypes("bool").columns
    df_enc[bc] = df_enc[bc].astype(int)
    return df_enc.reindex(columns=columns, fill_value=0)



#  HELPER — ZONE DE RISQUE 
def zone_risque(prob_sba1: float):
    """
    Retourne (niveau, couleur, message) selon la probabilité P(SBA=1).
    Trois zones :
      Vert    P ≥ 70 % => faible risque
      Orange  50 % ≤ P < 70 % => risque incertain
      Rouge   P < 50 % => risque élevé
    """
    if prob_sba1 >= 70:
        return "Faible risque",   "vert",   "🟢 Forte probabilité d'accouchement avec assistance qualifiée."
    elif prob_sba1 >= 50:
        return "Risque incertain", "orange", "🟡 La prédiction est 'Assistée' mais avec une marge étroite. Surveillance recommandée."
    else:
        return "Risque élevé",    "rouge",  "🔴 Risque élevé d'accouchement sans assistance qualifiée. Intervention prioritaire recommandée."



#  FONCTIONS DE PERSISTANCE  : stockage JSON

def load_history_from_file():
    """Charge l'historique depuis le fichier JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history_to_file(history):
    """Sauvegarde l'historique dans le fichier JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def delete_prediction_from_file(index):
    """Supprime une prédiction par son index (0-based dans la liste originale)."""
    history = load_history_from_file()
    if 0 <= index < len(history):
        history.pop(index)
        save_history_to_file(history)
        return True
    return False


#  STATE

if "page"        not in st.session_state: st.session_state.page = "accueil"
if "history"     not in st.session_state: 
    st.session_state.history = load_history_from_file()
if "last_result" not in st.session_state: st.session_state.last_result = None



#  SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2.5rem;margin-bottom:.4rem'>🤱</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                    font-weight:800;color:#F8FAFC'>SBA Prédicteur</div>
        <div style='font-size:.72rem;color:#64748B;margin-top:.2rem'>RDC · EDS 2023-2024</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    pages = [("","Accueil","accueil"),("","Formulaire","formulaire"),
             ("","Résultat","resultat"),("","Historique","historique")]
    for icon, label, key in pages:
        disabled = (key == "resultat" and st.session_state.last_result is None)
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     disabled=disabled, use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key; st.rerun()

    st.divider()
    thr_display = f"{THRESHOLD:.2f}" if MODEL_OK else "—"
    st.markdown(f"""
    <div style='font-size:.72rem;color:#475569;line-height:1.8'>
        <b style='color:#64748B'>Modèle :</b> Régression Logistique<br>
        <b style='color:#64748B'>class_weight :</b> balanced<br>
        <b style='color:#64748B'>Seuil optimal :</b> {thr_display}<br>
        <b style='color:#64748B'>Standardisation :</b> Aucune (variables catégorielles)<br>
        <b style='color:#64748B'>Prédictions :</b> {len(st.session_state.history)}<br><br>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_OK:
        st.divider()
        st.error("⚠️ Modèle absent.\nLance d'abord :\n```\npython train_model.py\n```")



#  PAGE 1 : ACCUEIL

if st.session_state.page == "accueil":

    st.markdown("""
    <div class='hero-card'>
        <div style='font-size:3.5rem;margin-bottom:1rem'>🤱</div>
        <h1 style='font-size:2.2rem;font-weight:800;
                   background:linear-gradient(135deg,#38BDF8,#818CF8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:.6rem'>SBA Prédicteur: L'Intelligence Artificielle au service de la santé maternelle en RDC.</h1>
        <p style='font-size:1rem;color:#94A3B8;max-width:600px;margin:0 auto 1.5rem;line-height:1.7'>
            Outil de prédiction du recours à un <strong style='color:#38BDF8'>personnel
            qualifié lors de l'accouchement</strong> chez les femmes mariées en République
            Démocratique du Congo, basé sur les données EDS-RDC 2023-2024.
        </p>
        <div style='display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap'>
            <span class='info-pill'> EDS-RDC 2023-2024</span>
            <span class='info-pill'> Régression Logistique</span>
            <span class='info-pill'> class_weight=balanced</span>
            <span class='info-pill'> AUC ≈ 0.83</span>
            <span class='info-pill'> N = 10 362 femmes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##  De quoi s'agit-il ?")
    col_ctx1, col_ctx2 = st.columns([3,2], gap="large")
    with col_ctx1:
        st.markdown("""
        La **mortalité maternelle** en République Démocratique du Congo reste parmi les
        plus élevées au monde. Une proportion significative des femmes accouchent encore
        **sans être accompagnées par un professionnel de santé qualifié**: médecin,
        infirmière ou sage-femme.

        Cette application utilise un **modèle de régression logistique** entraîné sur
        10 362 femmes mariées pour prédire si, selon son profil socio-démographique,
        une femme est susceptible d'accoucher avec ou sans assistance qualifiée.

        Le résultat s'affiche avec une **zone de risque colorée** (🟢 / 🟡 / 🔴)
        et les probabilités exactes pour chaque classe, afin d'aider les décideurs
        en santé publique à cibler les interventions.
        """)
    with col_ctx2:
        for val, label, color in [
            ("84.1 %","taux national d'assistées (EDS-RDC)","#34D399"),
            ("15.9 %","femmes sans assistance qualifiée","#F87171"),
            ("26 provinces","couverture nationale de l'analyse","#FBBF24"),
        ]:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size:2.2rem;font-weight:800;color:{color};
                            font-family:Space Grotesk,sans-serif'>{val}</div>
                <div style='font-size:.8rem;color:#64748B;margin-top:.3rem'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("##  Les facteurs les plus déterminants")
    f1,f2,f3,f4 = st.columns(4)
    for col, icon, titre, desc, c in [
        (f1,"","Consultations prénatales","4 CPN+ implique chances d'accouchement assisté ×9","#34D399"),
        (f2,"","Niveau d'instruction","Niveau supérieur implique 14× plus de chances","#38BDF8"),
        (f3,"","Niveau économique","Quintile riche implique 9× plus de chances","#818CF8"),
        (f4,"","Milieu de résidence","Rural implique 62 % de chances en moins","#FB923C"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card' style='border-color:{c}30'>
                <div style='font-size:1.8rem;margin-bottom:.5rem'>{icon}</div>
                <div style='font-size:.88rem;font-weight:700;color:{c};margin-bottom:.4rem'>{titre}</div>
                <div style='font-size:.78rem;color:#94A3B8;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("##  À qui s'adresse cette application ?")
    c1,c2,c3 = st.columns(3)
    for col, icon, titre, desc in [
        (c1,"","Décideurs en santé publique","Ministère de la Santé, ONG, agences OMS/UNICEF/UNFPA souhaitant cibler les interventions."),
        (c2,"","Chercheurs & épidémiologistes","Explorer les prédicteurs de l'accouchement assisté en RDC et tester différents profils."),
        (c3,"","Étudiants & enseignants","Illustration pédagogique du machine learning appliqué à la santé maternelle."),
    ]:
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div style='font-size:1.8rem;margin-bottom:.5rem'>{icon}</div>
                <div style='font-weight:700;color:#F1F5F9;margin-bottom:.3rem'>{titre}</div>
                <div style='font-size:.8rem;color:#94A3B8;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("## Comment utiliser l'application ?")
    s1,s2,s3,s4 = st.columns(4)
    for col, num, titre, desc in [
        (s1,"1","Remplir le formulaire","16 caractéristiques de la femme : âge, province, éducation, CPN, etc."),
        (s2,"2","Lancer la prédiction","Clique sur « Prédire » : le modèle analyse le profil instantanément."),
        (s3,"3","Lire le résultat","Classe prédite, probabilités et zone de risque colorée sur une page dédiée."),
        (s4,"4","Consulter l'historique","Toutes les prédictions sauvegardées et exportables en CSV."),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='width:36px;height:36px;border-radius:50%;
                            background:linear-gradient(135deg,#0EA5E9,#6366F1);
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;font-size:1rem;color:#fff;margin:0 auto .8rem'>{num}</div>
                <div style='font-weight:700;color:#F1F5F9;margin-bottom:.3rem;font-size:.9rem'>{titre}</div>
                <div style='font-size:.78rem;color:#94A3B8;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.info(" **Avertissement éthique** : Cet outil est une aide à la décision basée sur des données statistiques nationales. Il ne remplace pas un diagnostic médical individuel. Les prédictions reflètent des probabilités populationnelles et doivent être interprétées dans un contexte de politique de santé publique.")
    st.markdown("<br>", unsafe_allow_html=True)
    _,c_ctr,_ = st.columns([2,3,2])
    with c_ctr:
        if st.button("  Commencer a Remplir le formulaire", use_container_width=True):
            st.session_state.page = "formulaire"; st.rerun()



#  PAGE 2 : FORMULAIRE

elif st.session_state.page == "formulaire":

    st.markdown("#  Formulaire de prédiction")
    st.markdown("Remplis **tous les champs** ci-dessous. Chaque information est essentielle pour que le modèle puisse prédire correctement.")

    if not MODEL_OK:
        st.error(" Modèle non chargé. Lance d'abord `python train_model.py` dans le terminal.")
        st.stop()

    st.divider()

    with st.form("sba_form", clear_on_submit=False):

        st.markdown("###  Profil démographique")
        g1c1,g1c2,g1c3 = st.columns(3)
        with g1c1:
            age = st.selectbox("Groupe d'âge *",
                ["Sélectionner "]+list(MAP_AGE.keys()),
                help="Tranche d'âge de la femme mariée (15–49 ans)")
        with g1c2:
            residence = st.selectbox("Milieu de résidence *",
                ["Sélectionner","Milieu urbain","Milieu rural"],
                help="Zone géographique où vit la femme")
        with g1c3:
            region = st.selectbox("Province de résidence *",
                ["Sélectionner"]+OPTS_REGIONS,
                help="Province administrative de résidence")

        st.markdown("---")
        st.markdown("###  Éducation et situation économique")
        g2c1,g2c2 = st.columns(2)
        with g2c1:
            education = st.selectbox("Niveau d'instruction de la femme *",
                ["Sélectionner"]+list(MAP_EDUC.keys()),
                help="Dernier niveau d'études complété")
            richesse = st.selectbox("Niveau économique du ménage *",
                ["Sélectionner"]+list(MAP_RICHESSE.keys()),
                help="Quintile de richesse estimé du ménage")
        with g2c2:
            edu_mari = st.selectbox("Niveau d'instruction du mari *",
                ["Sélectionner"]+list(MAP_EDUC_MARI.keys()),
                help="Dernier niveau d'études complété par le conjoint")
            occupation = st.selectbox("Activité professionnelle de la femme *",
                ["Sélectionner"]+list(MAP_OCCUP.keys()),
                help="Principale activité exercée par la femme")

        st.markdown("---")
        st.markdown("###  Santé maternelle")
        g3c1,g3c2 = st.columns(2)
        with g3c1:
            cpn = st.selectbox("Consultations prénatales (CPN) *",
                ["Sélectionner"]+list(MAP_CPN.keys()),
                help="Nombre de visites prénatales lors de la dernière grossesse — l'OMS recommande ≥ 4")
        with g3c2:
            parite = st.selectbox("Nombre total d'enfants (parité) *",
                ["Sélectionner"]+list(MAP_PARITE.keys()),
                help="Nombre total d'enfants que la femme a eus jusqu'ici")

        st.markdown("---")
        st.markdown("### Accès à l'information et aux médias")
        g4c1,g4c2,g4c3 = st.columns(3)
        with g4c1:
            journal = st.selectbox("Lecture de journaux *",
                ["Sélectionner"]+OPTS_FREQ,
                help="Fréquence de lecture de la presse écrite")
        with g4c2:
            radio = st.selectbox("Écoute de la radio *",
                ["Sélectionner"]+OPTS_FREQ,
                help="Fréquence d'écoute de la radio")
        with g4c3:
            tv = st.selectbox("Visionnage de la télévision *",
                ["Sélectionner"]+OPTS_FREQ,
                help="Fréquence de visionnage de la télévision")

        st.markdown("---")
        st.markdown("### Autonomie décisionnelle dans le ménage")
        g5c1,g5c2,g5c3 = st.columns(3)
        with g5c1:
            dec_sante = st.selectbox("Qui décide des soins de santé ? *",
                ["Sélectionner"]+OPTS_DEC,
                help="Prise de décision concernant la santé personnelle")
        with g5c2:
            dec_achat = st.selectbox("Qui décide des grands achats ? *",
                ["Sélectionner"]+OPTS_DEC,
                help="Autonomie financière pour les dépenses importantes")
        with g5c3:
            dec_visite = st.selectbox("Qui décide des visites à la famille ? *",
                ["Sélectionner"]+OPTS_DEC,
                help="Liberté de déplacement de la femme")

        st.markdown("---")
        st.markdown("### Religion")
        religion = st.selectbox("Appartenance religieuse *",
            ["Sélectionner"]+list(MAP_RELIGION.keys()),
            help="Religion pratiquée par la femme")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("  Lancer la prédiction",
                                          use_container_width=True, type="primary")

    if submitted:
        fd = {"age":age,"residence":residence,"region":region,
              "education":education,"edu_mari":edu_mari,
              "richesse":richesse,"occupation":occupation,
              "cpn":cpn,"parite":parite,
              "journal":journal,"radio":radio,"tv":tv,
              "dec_sante":dec_sante,"dec_achat":dec_achat,"dec_visite":dec_visite,
              "religion":religion}
        manquants = [k for k,v in fd.items() if v.startswith("—")]
        if manquants:
            st.error(f" **{len(manquants)} champ(s) non rempli(s).** Merci de sélectionner une valeur pour chaque champ.")
        else:
            with st.spinner(" Analyse du profil en cours..."):
                X_in   = build_vector(fd, MODEL_COLS)
                # PAS DE SCALER — prédiction directe sur les 0/1
                proba  = model.predict_proba(X_in)[0]
                p1     = round(float(proba[1]) * 100, 1)
                p0     = round(float(proba[0]) * 100, 1)
                cls    = 1 if float(proba[1]) >= THRESHOLD else 0

            result = {**fd, "classe":cls, "prob_assistee":p1,
                      "prob_non_assistee":p0, "threshold":THRESHOLD,
                      "date":datetime.now().strftime("%d/%m/%Y à %H:%M")}
            st.session_state.last_result = result
            st.session_state.history.append(result)
            save_history_to_file(st.session_state.history)
            st.session_state.page = "resultat"
            st.rerun()



#  PAGE 3 : RÉSULTAT

elif st.session_state.page == "resultat":

    r = st.session_state.last_result
    if r is None:
        st.warning("Aucun résultat disponible.")
        if st.button("← Retour au formulaire"):
            st.session_state.page = "formulaire"; st.rerun()
        st.stop()

    is_ok = (r["classe"] == 1)

    # BANDEAU CLASSE PRÉDITE
    if is_ok:
        st.markdown("""
        <div class='result-main-yes'>
            <div style='font-size:3rem;margin-bottom:.5rem'>✅</div>
            <div style='font-family:Space Grotesk,sans-serif;font-size:1.9rem;
                        font-weight:800;color:#34D399;margin-bottom:.5rem'>
                Accouchement Assisté prédit
            </div>
            <div style='color:#86EFAC;font-size:.95rem;max-width:500px;margin:0 auto'>
                Le modèle prédit que cette femme accoucherait avec un
                <strong>personnel qualifié</strong> (médecin, infirmière ou sage-femme).
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-main-no'>
            <div style='font-size:3rem;margin-bottom:.5rem'>⚠️</div>
            <div style='font-family:Space Grotesk,sans-serif;font-size:1.9rem;
                        font-weight:800;color:#F87171;margin-bottom:.5rem'>
                Accouchement Non Assisté prédit
            </div>
            <div style='color:#FCA5A5;font-size:.95rem;max-width:500px;margin:0 auto'>
                Le modèle prédit un <strong>risque élevé d'accouchement sans assistance
                qualifiée</strong>. Ce profil mérite une attention prioritaire.
            </div>
        </div>""", unsafe_allow_html=True)

    #  ZONE DE RISQUE 
    st.markdown("###  Zone de risque")
    niveau, couleur, message = zone_risque(r["prob_assistee"])
    css_class = f"risk-box-{couleur}"
    couleur_hex = {"vert":"#34D399","orange":"#FBBF24","rouge":"#F87171"}[couleur]

    st.markdown(f"""
    <div class='{css_class}' style='margin-bottom:1rem'>
        <div style='font-size:1.1rem;font-weight:700;color:{couleur_hex};
                    margin-bottom:.4rem'>{niveau}</div>
        <div style='font-size:.9rem;color:#E2E8F0;line-height:1.6'>{message}</div>
        <div style='font-size:.75rem;color:#64748B;margin-top:.5rem'>
            Seuil de décision utilisé : {r.get("threshold", THRESHOLD):.2f}
            (optimisé pour maximiser la détection des non-assistées)
        </div>
    </div>
    """, unsafe_allow_html=True)

    #  PROBABILITÉS (solution 3 — affichage brut) 
    st.markdown("### Probabilités par classe")
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown(f"""
        <div class='stat-card' style='border-color:#34D39940'>
            <div style='font-size:.75rem;font-weight:700;color:#64748B;
                        text-transform:uppercase;letter-spacing:.5px;margin-bottom:.5rem'>
                ✅ Personnel qualifié (SBA = 1)
            </div>
            <div style='font-size:2.8rem;font-weight:800;color:#34D399;
                        font-family:Space Grotesk,sans-serif'>{r['prob_assistee']} %</div>
        </div>""", unsafe_allow_html=True)
        st.progress(r["prob_assistee"] / 100)

    with pc2:
        st.markdown(f"""
        <div class='stat-card' style='border-color:#F8717140'>
            <div style='font-size:.75rem;font-weight:700;color:#64748B;
                        text-transform:uppercase;letter-spacing:.5px;margin-bottom:.5rem'>
                ⚠️ Sans assistance (SBA = 0)
            </div>
            <div style='font-size:2.8rem;font-weight:800;color:#F87171;
                        font-family:Space Grotesk,sans-serif'>{r['prob_non_assistee']} %</div>
        </div>""", unsafe_allow_html=True)
        st.progress(r["prob_non_assistee"] / 100)

    st.divider()

    # PROFIL ANALYSÉ 
    st.markdown("###  Profil analysé")
    labels = {"age":"Groupe d'âge","residence":"Milieu de résidence","region":"Province",
              "education":"Instruction (femme)","edu_mari":"Instruction (mari)",
              "richesse":"Niveau économique","occupation":"Activité professionnelle",
              "cpn":"Consultations prénatales","parite":"Nombre d'enfants",
              "journal":"Lecture journaux","radio":"Écoute radio","tv":"Visionnage TV",
              "dec_sante":"Décision soins santé","dec_achat":"Décision grands achats",
              "dec_visite":"Décision visites famille","religion":"Religion"}
    rows = [{"Variable":labels[k],"Valeur renseignée":r[k]}
            for k in labels if k in r and r[k]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f" Prédit le {r['date']} · Modèle : Régression Logistique (class_weight=balanced) · Seuil = {r.get('threshold',THRESHOLD):.2f}")

    st.divider()
    nb1,nb2 = st.columns(2)
    with nb1:
        if st.button("  Nouvelle prédiction", use_container_width=True):
            st.session_state.page = "formulaire"; st.rerun()
    with nb2:
        if st.button(" Voir l'historique", use_container_width=True):
            st.session_state.page = "historique"; st.rerun()



#  PAGE 4 : HISTORIQUE

elif st.session_state.page == "historique":

    st.markdown("# Historique des prédictions")
    hist = st.session_state.history
    n    = len(hist)

    if n == 0:
        st.info("Aucune prédiction enregistrée dans cette session.")
        if st.button("  Faire la première prédiction", use_container_width=True):
            st.session_state.page = "formulaire"; st.rerun()
        st.stop()

    nb_ok  = sum(1 for h in hist if h["classe"] == 1)
    nb_non = n - nb_ok
    moy_p  = round(sum(h["prob_assistee"] for h in hist) / n, 1)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total prédictions", n)
    m2.metric("✅ Assistées",       nb_ok,  delta=f"{round(nb_ok/n*100)} %")
    m3.metric("⚠️ Non assistées",  nb_non, delta=f"{round(nb_non/n*100)} %", delta_color="inverse")
    m4.metric("Prob. moy. SBA=1",  f"{moy_p} %")

    st.divider()
    st.markdown("### Toutes les prédictions de la session")

    rows = []
    for i, h in enumerate(reversed(hist)):
        niv, _, _ = zone_risque(h["prob_assistee"])
        rows.append({
            "#":              n - i,
            "Date":           h["date"],
            "Province":       h["region"],
            "Âge":            h["age"],
            "Résidence":      h["residence"],
            "Éducation femme": h["education"],
            "Éducation mari": h["edu_mari"],
            "Richesse":       h["richesse"],
            "Activité":       h["occupation"],
            "CPN":            h["cpn"],
            "Parité":         h["parite"],
            "Journal":        h["journal"],
            "Radio":          h["radio"],
            "TV":             h["tv"],
            "Décision santé": h["dec_sante"],
            "Décision achats": h["dec_achat"],
            "Décision visites": h["dec_visite"],
            "Religion":       h["religion"],
            "Classe prédite": "✅ Assistée (1)" if h["classe"]==1 else "⚠️ Non assistée (0)",
            "Zone de risque": niv,
            "Prob. SBA=1 (%)": h["prob_assistee"],
            "Prob. SBA=0 (%)": h["prob_non_assistee"],
        })

    df_h = pd.DataFrame(rows)
    st.dataframe(df_h, use_container_width=True, hide_index=True,
        column_config={
            "Prob. SBA=1 (%)": st.column_config.ProgressColumn(
                "Prob. SBA=1 (%)", min_value=0, max_value=100, format="%.1f%%"),
            "Prob. SBA=0 (%)": st.column_config.ProgressColumn(
                "Prob. SBA=0 (%)", min_value=0, max_value=100, format="%.1f%%"),
        })

    csv = df_h.to_csv(index=False).encode("utf-8")
    st.download_button(" Télécharger l'historique (CSV)", data=csv,
        file_name=f"sba_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv", use_container_width=True)

    st.divider()
    # SUPPRESSION INDIVIDUELLE 
    st.markdown("### Supprimer une prédiction")
    st.info("Sélectionne le numéro de la prédiction à supprimer, puis clique sur le bouton rouge.")

    del_col1, del_col2, del_col3 = st.columns([2, 2, 3])
    with del_col1:
        del_idx = st.number_input("Numéro à supprimer :", 1, n, 1, 1, key="del_idx")
    with del_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Supprimer cette prédiction", type="primary", use_container_width=True):
            original_idx = n - del_idx
            if 0 <= original_idx < len(hist):
                deleted = hist.pop(original_idx)
                st.session_state.history = hist
                save_history_to_file(hist)
                st.success(f"✅ Prédiction #{del_idx} supprimée avec succès.")
                st.rerun()
            else:
                st.error("❌ Numéro invalide.")
    with del_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Vider tout l'historique", type="secondary", use_container_width=True):
            st.session_state.history = []
            save_history_to_file([])
            st.success("✅ Tout l'historique a été supprimé.")
            st.rerun()

    st.divider()

    # DÉTAIL D'UNE ENTRÉE 
    with st.expander(" Voir le détail complet d'une prédiction"):
        idx = st.number_input("Numéro :", 1, n, n, 1)
        h   = hist[idx - 1]
        niv, col, msg = zone_risque(h["prob_assistee"])
        d1,d2 = st.columns(2)
        with d1:
            if h["classe"]==1: st.success(f"✅ Assistée (SBA = 1)")
            else:               st.error(f"⚠️ Non assistée (SBA = 0)")
            st.markdown(f"**Zone de risque :** {niv}")
            st.metric("Prob. SBA=1", f"{h['prob_assistee']} %")
            st.progress(h["prob_assistee"]/100)
            st.metric("Prob. SBA=0", f"{h['prob_non_assistee']} %")
            st.progress(h["prob_non_assistee"]/100)
        with d2:
            for k,label in [("age","Âge"),("residence","Résidence"),("region","Province"),
                            ("education","Instruction femme"),("edu_mari","Instruction mari"),
                            ("richesse","Richesse"),("occupation","Activité professionnelle"),
                            ("cpn","CPN"),("parite","Parité"),
                            ("journal","Lecture journaux"),("radio","Écoute radio"),("tv","Visionnage TV"),
                            ("dec_sante","Décision soins santé"),("dec_achat","Décision grands achats"),
                            ("dec_visite","Décision visites famille"),("religion","Religion")]:
                st.text(f"{label} : {h.get(k,'—')}")
            st.caption(f"Prédit le {h.get('date','—')} · Seuil = {h.get('threshold',THRESHOLD):.2f}")

    if st.button(" Nouvelle prédiction", use_container_width=True):
        st.session_state.page = "formulaire"; st.rerun()
