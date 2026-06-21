"""
app.py — Application Streamlit : Prédiction de l'accouchement assisté (SBA) au Cameroun
Auteur : NTANTAME DIHEWOU TED HARRIS — Master 1 Data Science, IUSJ
Données : EDS Cameroun 2018 (CMIR71FL)
Modèle : Random Forest | AUC ≈ 0.90
Seuil : Optimal (0.52)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
from datetime import datetime

# CONSTANTES
HISTORY_FILE = "sba_history.json"

# CONFIG PAGE
st.set_page_config(
    page_title="SBA Prédicteur Cameroun",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS GLOBAL (inchangé)
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
.risk-box-rouge  { background:#2D0A0A; border:1.5px solid #F87171; border-radius:12px; padding:1rem 1.2rem; }
.risk-box-orange { background:#1C1200; border:1.5px solid #FBBF24; border-radius:12px; padding:1rem 1.2rem; }
.risk-box-vert   { background:#052E16; border:1.5px solid #34D399; border-radius:12px; padding:1rem 1.2rem; }
.info-pill { display:inline-block; background:#0EA5E915; color:#38BDF8; border:1px solid #0EA5E940; border-radius:20px; padding:4px 14px; font-size:.78rem; font-weight:600; margin:3px; }
</style>
""", unsafe_allow_html=True)

# CHARGEMENT DU MODÈLE
@st.cache_resource(show_spinner=False)
def load_model():
    base = "model_files"
    try:
        m = joblib.load(os.path.join(base, "model.pkl"))
        c = joblib.load(os.path.join(base, "columns.pkl"))
        thr = joblib.load(os.path.join(base, "threshold.pkl"))
        return m, c, thr, True
    except:
        return None, None, 0.5, False

model, MODEL_COLS, THRESHOLD, MODEL_OK = load_model()

# ====================== MAPPINGS CAMEROUN ======================
MAP_AGE = {"15–19 ans":1,"20–24 ans":2,"25–29 ans":3,"30–34 ans":4,
           "35–39 ans":5,"40–44 ans":6,"45–49 ans":7}

MAP_RESIDENCE = {"Milieu urbain":1,"Milieu rural":2}

MAP_REGION = {
    "Adamaoua":1, "Centre (sans Yaounde)":2, "Douala":3, "Est":4,
    "Extreme-Nord":5, "Littoral (sans Douala)":6, "Nord":7,
    "Nord-Ouest":8, "Ouest":9, "Sud":10, "Sud-Ouest":11, "Yaounde":12
}

MAP_EDUC = {"Aucune instruction":0,"Niveau primaire":1,
            "Niveau secondaire":2,"Niveau supérieur (université)":3}

MAP_EDUC_MARI = {"Aucune instruction":0,"Niveau primaire":1,
                 "Niveau secondaire":2,"Niveau supérieur (université)":3,"Ne sait pas":8}

MAP_RICHESSE = {"Très pauvre":1,"Pauvre":2,"Revenu moyen":3,"Riche":4,"Très Riche":5}

MAP_CPN = {"Aucune visite":"0", "Au moins 1 visite":"1"}   # Adapté à tes 2 modalités

MAP_PARITE = {"1 à 2 enfants":"1","3 à 4 enfants":"2","5 enfants ou plus":"3"}
MAP_AUTONOMIE = {"Basse":1, "Moyenne":2, "Haute":3}

MAP_FREQ = {"Non":0, "Oui":1}   # Pour Media_exposure (Oui/Non)
MAP_VIOLENCE_CONJ={"Rejette la violence":"1","Justifie la violence":"0"}  # référence=1=Rejette dans le modèle
MAP_RELIGION = {
    "Catholique":1, 
    "Protestant":2, 
    "Autre chrétien":3, 
    "Musulman":4, 
    "Animiste":5, 
    "Aucun":7, 
    "Autre":96
}

OPTS_FREQ = list(MAP_FREQ.keys())
OPTS_DEC = list(MAP_AUTONOMIE.keys())
OPTS_REGIONS = list(MAP_REGION.keys())

# ====================== BUILD VECTOR ======================
def build_vector(fd: dict, columns: list) -> pd.DataFrame:
    # Gestion autonomie : peut être string ou déjà un int
    autonomie_val = fd.get("autonomie", 2)
    if isinstance(autonomie_val, str):
        autonomie_val = MAP_AUTONOMIE[autonomie_val]
    
    # Gestion violence : peut être string ou déjà un int  
    violence_val = fd.get("violence", 0)
    if isinstance(violence_val, str):
        violence_val = int(MAP_VIOLENCE_CONJ[violence_val])
    
    raw = {
        "V013": MAP_AGE[fd["age"]],
        "V024": MAP_REGION[fd["region"]],
        "V025": MAP_RESIDENCE[fd["residence"]],
        "V106": MAP_EDUC[fd["education"]],
        "V701": MAP_EDUC_MARI[fd["edu_mari"]],
        "V130": MAP_RELIGION.get(fd["religion"], 96),
        "V190": MAP_RICHESSE[fd["richesse"]],
        "Media_exposure": MAP_FREQ.get(fd["media"], 0),
        "autonomie_decisionnelle": autonomie_val,   # ← nombre 1, 2 ou 3
        "attitude_violence": violence_val,            # ← nombre 0 ou 1
        "cpn_cat": MAP_CPN.get(fd["cpn"], "0"),
        "parite": MAP_PARITE[fd["parite"]],
    }
    df = pd.DataFrame([raw])

    # Forcer les catégories
    df["V013"] = pd.Categorical(df["V013"], categories=[1,2,3,4,5,6,7])
    df["V024"] = pd.Categorical(df["V024"], categories=[5,1,2,3,4,6,7,8,9,10,11,12])  # référence=5=Extrême-Nord, comme dans le modèle
    df["V025"] = pd.Categorical(df["V025"], categories=[2,1])
    df["V106"] = pd.Categorical(df["V106"], categories=[0,1,2,3])
    df["V701"] = pd.Categorical(df["V701"], categories=[0,1,2,3,8])
    df["V130"] = pd.Categorical(df["V130"], categories=[4,1,2,3,5,7,96])  # référence=4=Musulman, comme dans le modèle
    df["V190"] = pd.Categorical(df["V190"], categories=[1,2,3,4,5])
    df["Media_exposure"] = pd.Categorical(df["Media_exposure"], categories=[0,1])
    df["autonomie_decisionnelle"] = pd.Categorical(df["autonomie_decisionnelle"], categories=[1,2,3])
    df["attitude_violence"] = pd.Categorical(df["attitude_violence"], categories=[1,0])  # référence=1=Rejette, comme dans le modèle
    df["cpn_cat"] = pd.Categorical(df["cpn_cat"], categories=["0","1"])
    df["parite"] = pd.Categorical(df["parite"], categories=["3","2","1"])

    vars_cat = ["V013","V024","V025","V106","V130","V190","V701","Media_exposure",
                "autonomie_decisionnelle","attitude_violence","parite","cpn_cat"]

    df_enc = pd.get_dummies(df, columns=vars_cat, drop_first=True)
    bc = df_enc.select_dtypes("bool").columns
    df_enc[bc] = df_enc[bc].astype(int)
    # DEBUG : compare les colonnes
    print("Colonnes générées :", sorted(df_enc.columns.tolist()))
    print("Colonnes attendues :", sorted(columns))
    print("Manquantes :", set(columns) - set(df_enc.columns))
    print("En trop :", set(df_enc.columns) - set(columns))
    return df_enc.reindex(columns=columns, fill_value=0)

# ====================== ZONE DE RISQUE ======================
def zone_risque(prob_sba1: float, classe: int = 1):
    """
    Cohérence garantie entre classe prédite et zone de risque :
    - Classe 0 (Non Assistée) → toujours Risque élevé
    - Classe 1 (Assistée)     → graduer selon la probabilité
        P >= 85 % → Faible risque
        P <  85 % → Risque modéré (assistée mais de justesse)
    """
    if classe == 0:
        return ("Risque élevé", "rouge",
                "🔴 Risque élevé d'accouchement sans assistance qualifiée. Intervention prioritaire recommandée.")
    else:
        if prob_sba1 >= 70:
            return ("Faible risque", "vert",
                    "🟢 Forte probabilité d'accouchement avec assistance qualifiée.")
        else:
            return ("Risque modéré", "orange",
                    f"🟡 La femme est prédite assistée, mais la probabilité reste modérée ({prob_sba1} %). Suivi recommandé.")

# ====================== HISTORY ======================
def load_history_from_file():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history_to_file(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ====================== STATE ======================
if "page" not in st.session_state: st.session_state.page = "accueil"
if "history" not in st.session_state: 
    st.session_state.history = load_history_from_file()
if "last_result" not in st.session_state: st.session_state.last_result = None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2.5rem;margin-bottom:.4rem'>🤱</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                    font-weight:800;color:#F8FAFC'>SBA Prédicteur</div>
        <div style='font-size:.72rem;color:#64748B;margin-top:.2rem'>Cameroun · EDS 2018</div>
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
        <b style='color:#64748B'>Modèle :</b> Random Forest<br>
        <b style='color:#64748B'>Seuil optimal :</b> {thr_display}<br>
        <b style='color:#64748B'>Prédictions :</b> {len(st.session_state.history)}<br><br>
    </div>
    """, unsafe_allow_html=True)

# ====================== PAGE 1 : ACCUEIL ======================
if st.session_state.page == "accueil":

    st.markdown("""
    <div class='hero-card'>
        <div style='font-size:3.5rem;margin-bottom:1rem'>🤱</div>
        <h1 style='font-size:2.2rem;font-weight:800;
                   background:linear-gradient(135deg,#38BDF8,#818CF8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:.6rem'>SBA Prédicteur: L'Intelligence Artificielle au service de la santé maternelle au Cameroun.</h1>
        <p style='font-size:1rem;color:#94A3B8;max-width:600px;margin:0 auto 1.5rem;line-height:1.7'>
            Outil de prédiction du recours à un <strong style='color:#38BDF8'>personnel
            qualifié lors de l'accouchement</strong> chez les femmes mariées au Cameroun, 
            basé sur les données EDS Cameroun 2018 (CMIR71FL).
        </p>
        <div style='display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap'>
            <span class='info-pill'> EDS Cameroun 2018</span>
            <span class='info-pill'> Random Forest</span>
            <span class='info-pill'> AUC ≈ 0.90</span>
            <span class='info-pill'> N = 5013 femmes mariées</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## De quoi s'agit-il ?")
    col_ctx1, col_ctx2 = st.columns([3,2], gap="large")
    with col_ctx1:
        st.markdown("""
        La **mortalité maternelle** au Cameroun reste un défi majeur de santé publique. 
        Une proportion importante des femmes accouchent encore **sans assistance d’un personnel de santé qualifié** 
        (médecin, infirmière ou sage-femme).

        Cette application utilise un **modèle Random Forest** entraîné sur 5013 femmes mariées 
        pour prédire, selon son profil socio-démographique, la probabilité qu’une femme accouche 
        avec ou sans assistance qualifiée.

        Le résultat s’affiche avec une **zone de risque colorée** (🟢 / 🟡 / 🔴) 
        et les probabilités exactes pour aider à mieux cibler les interventions.
        """)
    with col_ctx2:
        for val, label, color in [
            ("68.1 %","taux national d'assistées (EDS 2018)","#34D399"),
            ("31.9 %","femmes sans assistance qualifiée","#F87171"),
            ("12 régions","couverture nationale de l'analyse","#FBBF24"),
        ]:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size:2.2rem;font-weight:800;color:{color};
                            font-family:Space Grotesk,sans-serif'>{val}</div>
                <div style='font-size:.8rem;color:#64748B;margin-top:.3rem'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("## Les facteurs les plus déterminants")
    f1,f2,f3,f4 = st.columns(4)
    for col, icon, titre, desc, c in [
        (f1,"","Consultations prénatales","Au moins 1 CPN augmente fortement les chances","#34D399"),
        (f2,"","Exposition aux médias","Être exposée aux médias est un levier important","#38BDF8"),
        (f3,"","Milieu de résidence","Vivre en milieu urbain augmente les chances","#818CF8"),
        (f4,"","Niveau d'éducation","Éducation secondaire de la femme est très déterminant","#FB923C"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card' style='border-color:{c}30'>
                <div style='font-size:1.8rem;margin-bottom:.5rem'>{icon}</div>
                <div style='font-size:.88rem;font-weight:700;color:{c};margin-bottom:.4rem'>{titre}</div>
                <div style='font-size:.78rem;color:#94A3B8;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("## À qui s'adresse cette application ?")
    c1,c2,c3 = st.columns(3)
    for col, icon, titre, desc in [
        (c1,"","Décideurs en santé publique","Ministère de la Santé, ONG, agences OMS/UNICEF/UNFPA souhaitant cibler les interventions."),
        (c2,"","Chercheurs & épidémiologistes","Explorer les prédicteurs de l'accouchement assisté au Cameroun et tester différents profils."),
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
        (s1,"1","Remplir le formulaire","Caractéristiques de la femme : âge, région, éducation, CPN, etc."),
        (s2,"2","Lancer la prédiction","Clique sur « Prédire » : le modèle analyse le profil instantanément."),
        (s3,"3","Lire le résultat","Classe prédite, probabilités et zone de risque colorée."),
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
        if st.button(" Commencer à Remplir le formulaire", use_container_width=True):
            st.session_state.page = "formulaire"; st.rerun()

# ====================== PAGE 2 : FORMULAIRE ======================
elif st.session_state.page == "formulaire":

    st.markdown("# Formulaire de prédiction")
    st.markdown("Remplis **tous les champs** ci-dessous. Chaque information est essentielle pour que le modèle puisse prédire correctement.")

    if not MODEL_OK:
        st.error(" Modèle non chargé. Lance d'abord `python train_model_RF.py` dans le terminal.")
        st.stop()

    st.divider()

    with st.form("sba_form", clear_on_submit=False):

        st.markdown("### Profil démographique")
        g1c1,g1c2,g1c3 = st.columns(3)
        with g1c1:
            age = st.selectbox("Groupe d'âge *", ["Sélectionner "] + list(MAP_AGE.keys()))
        with g1c2:
            residence = st.selectbox("Milieu de résidence *", ["Sélectionner"] + list(MAP_RESIDENCE.keys()))
        with g1c3:
            region = st.selectbox("Région de résidence *", ["Sélectionner"] + OPTS_REGIONS)

        st.markdown("---")
        st.markdown("### Éducation et situation économique")
        g2c1,g2c2 = st.columns(2)
        with g2c1:
            education = st.selectbox("Niveau d'instruction de la femme *", ["Sélectionner"] + list(MAP_EDUC.keys()))
            richesse = st.selectbox("Quintile de richesse du ménage *", ["Sélectionner"] + list(MAP_RICHESSE.keys()))
        with g2c2:
            edu_mari = st.selectbox("Niveau d'instruction du mari *", ["Sélectionner"] + list(MAP_EDUC_MARI.keys()))

        st.markdown("---")
        st.markdown("### Santé maternelle")
        g3c1,g3c2 = st.columns(2)
        with g3c1:
            cpn = st.selectbox("Consultations prénatales (ANC) *", 
                ["Sélectionner", "Aucune visite", "Au moins 1 visite"])
        with g3c2:
            parite = st.selectbox("Nombre total d'enfants (parité) *", ["Sélectionner"] + list(MAP_PARITE.keys()))

        st.markdown("---")
        st.markdown("### Exposition aux médias, autonomie et attitude")
        g4c1,g4c2,g4c3 = st.columns(3)
        with g4c1:
            media = st.selectbox("Exposition aux médias *", ["Sélectionner", "Non", "Oui"])
        with g4c2:
            autonomie = st.selectbox("Niveau d'autonomie décisionnelle *", 
                ["Sélectionner", "Basse", "Moyenne", "Haute"])
        with g4c3:
            violence = st.selectbox("Attitude envers la violence conjugale *", 
                ["Sélectionner", "Rejette la violence", "Justifie la violence"])

        st.markdown("---")
        st.markdown("### Religion")
        religion = st.selectbox("Appartenance religieuse *", 
            ["Sélectionner", "Catholique", "Protestant", "Autre chrétien", 
             "Musulman", "Animiste", "Aucun", "Autre"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(" Lancer la prédiction",
                                          use_container_width=True, type="primary")

    if submitted:
        # Conversion des labels en valeurs numériques pour le modèle
        # Stocker les labels lisibles pour l'affichage
        fd = {
            "age": age,
            "residence": residence,
            "region": region,
            "education": education,
            "edu_mari": edu_mari,
            "richesse": richesse,
            "cpn": cpn,
            "parite": parite,
            "media": media,
            "autonomie": autonomie,         # label lisible : "Basse"/"Moyenne"/"Haute"
            "violence": violence,           # label lisible : "Rejette..."/"Justifie..."
            "religion": religion
        }
        # Valeurs numériques séparées uniquement pour le modèle
        fd_model = dict(fd)
        fd_model["autonomie"] = {"Basse":1, "Moyenne":2, "Haute":3}.get(autonomie, 2)
        fd_model["violence"]  = {"Rejette la violence":1, "Justifie la violence":0}.get(violence, 1)  # référence=1=Rejette

        manquants = [k for k,v in fd.items() if isinstance(v, str) and v.startswith("Sélectionner")]
        if manquants:
            st.error(f" **{len(manquants)} champ(s) non rempli(s).** Merci de sélectionner une valeur pour chaque champ.")
        else:
            with st.spinner(" Analyse du profil en cours..."):
                X_in = build_vector(fd_model, MODEL_COLS)
                proba = model.predict_proba(X_in)[0]
                p1 = round(float(proba[1]) * 100, 1)
                p0 = round(float(proba[0]) * 100, 1)
                cls = 1 if float(proba[1]) >= THRESHOLD else 0

            result = {**fd, "classe":cls, "prob_assistee":p1,
                      "prob_non_assistee":p0, "threshold":THRESHOLD,
                      "date":datetime.now().strftime("%d/%m/%Y à %H:%M")}
            
            st.session_state.last_result = result
            st.session_state.history.append(result)
            save_history_to_file(st.session_state.history)
            st.session_state.page = "resultat"
            st.rerun()

# ====================== PAGE 3 : RÉSULTAT ======================
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
                Le modèle prédit que cette femme accoucherait avec un <strong>personnel qualifié</strong>.
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
                Le modèle prédit un <strong>risque élevé d'accouchement sans assistance qualifiée</strong>.
            </div>
        </div>""", unsafe_allow_html=True)

    # ZONE DE RISQUE
    st.markdown("### Zone de risque")
    niveau, couleur, message = zone_risque(r["prob_assistee"], r["classe"])
    css_class = f"risk-box-{couleur}"
    couleur_hex = {"vert":"#34D399","orange":"#FBBF24","rouge":"#F87171"}[couleur]

    st.markdown(f"""
    <div class='{css_class}' style='margin-bottom:1rem'>
        <div style='font-size:1.1rem;font-weight:700;color:{couleur_hex};
                    margin-bottom:.4rem'>{niveau}</div>
        <div style='font-size:.9rem;color:#E2E8F0;line-height:1.6'>{message}</div>
        <div style='font-size:.75rem;color:#64748B;margin-top:.5rem'>
            Seuil de décision utilisé : {r.get("threshold", THRESHOLD):.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PROBABILITÉS
    st.markdown("### Probabilités par classe")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(f"""
        <div class='stat-card' style='border-color:#34D39940'>
            <div style='font-size:.75rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.5rem'>
                ✅ Personnel qualifié (SBA = 1)
            </div>
            <div style='font-size:2.8rem;font-weight:800;color:#34D399;font-family:Space Grotesk,sans-serif'>
                {r.get('prob_assistee', 0)} %
            </div>
        </div>""", unsafe_allow_html=True)
        st.progress(r.get("prob_assistee", 0) / 100)

    with pc2:
        st.markdown(f"""
        <div class='stat-card' style='border-color:#F8717140'>
            <div style='font-size:.75rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.5rem'>
                ⚠️ Sans assistance (SBA = 0)
            </div>
            <div style='font-size:2.8rem;font-weight:800;color:#F87171;font-family:Space Grotesk,sans-serif'>
                {r.get('prob_non_assistee', 0)} %
            </div>
        </div>""", unsafe_allow_html=True)
        st.progress(r.get("prob_non_assistee", 0) / 100)

    st.divider()

    # PROFIL ANALYSÉ
    st.markdown("### Profil analysé")
    labels = {
        "age":"Groupe d'âge",
        "residence":"Milieu de résidence",
        "region":"Région",
        "education":"Niveau d'instruction femme",
        "edu_mari":"Niveau d'instruction mari",
        "richesse":"Quintile de richesse",
        "cpn":"Consultations prénatales",
        "parite":"Parité",
        "media":"Exposition aux médias",
        "autonomie":"Autonomie décisionnelle",
        "violence":"Attitude violence",
        "religion":"Religion"
    }

    rows = [{"Variable":labels.get(k, k), "Valeur renseignée":r.get(k, "—")} 
            for k in labels if k in r]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.caption(f" Prédit le {r.get('date','—')} · Modèle : Random Forest · Seuil = {r.get('threshold', THRESHOLD):.2f}")

    st.divider()
    nb1,nb2 = st.columns(2)
    with nb1:
        if st.button(" Nouvelle prédiction", use_container_width=True):
            st.session_state.page = "formulaire"; st.rerun()
    with nb2:
        if st.button(" Voir l'historique", use_container_width=True):
            st.session_state.page = "historique"; st.rerun()



# ====================== PAGE 4 : HISTORIQUE ======================
elif st.session_state.page == "historique":

    st.markdown("# Historique des prédictions")
    hist = st.session_state.history
    n = len(hist)

    if n == 0:
        st.info("Aucune prédiction enregistrée dans cette session.")
        if st.button(" Faire la première prédiction", use_container_width=True):
            st.session_state.page = "formulaire"; st.rerun()
        st.stop()

    nb_ok = sum(1 for h in hist if h["classe"] == 1)
    nb_non = n - nb_ok
    moy_p = round(sum(h["prob_assistee"] for h in hist) / n, 1)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total prédictions", n)
    m2.metric("✅ Assistées", nb_ok, delta=f"{round(nb_ok/n*100)} %")
    m3.metric("⚠️ Non assistées", nb_non, delta=f"{round(nb_non/n*100)} %", delta_color="inverse")
    m4.metric("Prob. moy. SBA=1", f"{moy_p} %")

    st.divider()
    st.markdown("### Toutes les prédictions de la session")

    rows = []
    for i, h in enumerate(reversed(hist)):
        niv, _, _ = zone_risque(h["prob_assistee"], h["classe"])
        rows.append({
            "#": n - i,
            "Date": h["date"],
            "Région": h.get("region", "—"),
            "Âge": h.get("age", "—"),
            "Résidence": h.get("residence", "—"),
            "Éducation femme": h.get("education", "—"),
            "Éducation mari": h.get("edu_mari", "—"),
            "Richesse": h.get("richesse", "—"),
            "CPN": h.get("cpn", "—"),
            "Parité": h.get("parite", "—"),
            "Médias": h.get("media", "—"),
            "Autonomie": h.get("autonomie", "—"),
            "Violence": h.get("violence", "—"),
            "Religion": h.get("religion", "—"),
            "Classe prédite": "✅ Assistée (1)" if h["classe"]==1 else "⚠️ Non assistée (0)",
            "Zone de risque": niv,
            "Prob. SBA=1 (%)": h.get("prob_assistee", 0),
            "Prob. SBA=0 (%)": h.get("prob_non_assistee", 0),
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
    # SUPPRESSION
    st.markdown("### Supprimer une prédiction")
    st.info("Sélectionne le numéro de la prédiction à supprimer.")
    del_col1, del_col2, del_col3 = st.columns([2, 2, 3])
    with del_col1:
        del_idx = st.number_input("Numéro à supprimer :", 1, n, 1, 1, key="del_idx")
    with del_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Supprimer cette prédiction", type="primary", use_container_width=True):
            original_idx = n - del_idx
            if 0 <= original_idx < len(hist):
                hist.pop(original_idx)
                st.session_state.history = hist
                save_history_to_file(hist)
                st.success(f"✅ Prédiction #{del_idx} supprimée.")
                st.rerun()
            else:
                st.error("❌ Numéro invalide.")
    with del_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(" Vider tout l'historique", type="secondary", use_container_width=True):
            st.session_state.history = []
            save_history_to_file([])
            st.success("✅ Historique vidé.")
            st.rerun()

    st.divider()
    # DÉTAIL
    with st.expander(" Voir le détail complet d'une prédiction"):
        idx = st.number_input("Numéro :", 1, n, n, 1)
        h = hist[idx - 1]
        niv, col, msg = zone_risque(h["prob_assistee"], h["classe"])
        d1,d2 = st.columns(2)
        with d1:
            if h["classe"]==1: st.success("✅ Assistée (SBA = 1)")
            else: st.error("⚠️ Non assistée (SBA = 0)")
            st.metric("Prob. SBA=1", f"{h.get('prob_assistee',0)} %")
            st.progress(h.get("prob_assistee",0)/100)
            st.metric("Prob. SBA=0", f"{h.get('prob_non_assistee',0)} %")
            st.progress(h.get("prob_non_assistee",0)/100)
        with d2:
            for k,label in [("age","Âge"),("residence","Résidence"),("region","Région"),
                            ("education","Éducation femme"),("edu_mari","Éducation mari"),
                            ("richesse","Richesse"),("cpn","CPN"),("parite","Parité"),
                            ("media","Médias"),("autonomie","Autonomie"),
                            ("violence","Attitude violence"),("religion","Religion")]:
                st.text(f"{label} : {h.get(k,'—')}")
            st.caption(f"Prédit le {h.get('date','—')} · Seuil = {h.get('threshold',THRESHOLD):.2f}")

    if st.button(" Nouvelle prédiction", use_container_width=True):
        st.session_state.page = "formulaire"; st.rerun()
