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
import joblib, os
from datetime import datetime


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
.risk-box-rouge   { background:#2D0A0A; border:1.5px solid #F87171; border-radius:12px; padding:1rem 1.2rem; }
.risk-box-orange { background:#1C1200; border:1.5px solid #FBBF24; border-radius:12px; padding:1rem 1.2rem; }
.risk-box-vert   { background:#052E16; border:1.5px solid #34D399; border-radius:12px; padding:1rem 1.2rem; }
.info-pill { display:inline-block; background:#0EA5E915; color:#38BDF8; border:1px solid #0EA5E940; border-radius:20px; padding:4px 14px; font-size:.78rem; font-weight:600; margin:3px; }
</style>
""", unsafe_allow_html=True)


#  CHARGEMENT DU MODÈLE + SEUIL
@st.cache_resource(show_spinner=False)
def load_model():
    # REMARQUE : Ajustez la variable 'base' si vos fichiers .pkl sont stockés dans un dossier spécifique sur GitHub
    base = "" # Laissez vide "" si les fichiers sont directement à la racine de votre GitHub
    try:
        m   = joblib.load(os.path.join(base, "model.pkl")) if base else joblib.load("model.pkl")
        c   = joblib.load(os.path.join(base, "columns.pkl")) if base else joblib.load("columns.pkl")
        thr = joblib.load(os.path.join(base, "threshold.pkl")) if base else joblib.load("threshold.pkl")
        return m, c, thr, True
    except Exception as e:
        return None, None, 0.5, False

model, MODEL_COLS, THRESHOLD, MODEL_OK = load_model()


#  MAPPINGS LISIBLES → CODES EDS
MAP_AGE       = {"15–19 ans":1,"20–24 ans":2,"25–29 ans":3,"30–34 ans":4,"35–39 ans":5,"40–44 ans":6,"45–49 ans":7}
MAP_RESIDENCE = {"Milieu urbain":1,"Milieu rural":2}
MAP_REGION    = {"Kinshasa":1,"Kongo Central":2,"Kwango":3,"Kwilu":4,"Mai-Ndombe":5,"Équateur":6,"Nord Ubangi":7,"Sud Ubangi":8,"Mongala":9,"Tshuapa":10,"Tshopo":11,"Bas-Uele":12,"Haut Uele":13,"Ituri":14,"Nord-Kivu":15,"Sud-Kivu":16,"Maniema":17,"Haut-Katanga":18,"Haut Lomami":19,"Lualaba":20,"Tanganyika":21,"Lomami":22,"Sankuru":23,"Kasaï Oriental":24,"Kasaï":25,"Kasaï Central":26}
MAP_EDUC      = {"Aucune instruction":0,"Niveau primaire":1,"Niveau secondaire":2,"Niveau supérieur (université)":3}
MAP_EDUC_MARI = {"Aucune instruction":0,"Niveau primaire":1,"Niveau secondaire":2,"Niveau supérieur (université)":3,"Ne sait pas":8}
MAP_RICHESSE  = {"Très pauvre":1,"Pauvre":2,"Revenu moyen":3,"Aisé":4,"Très aisé":5}
MAP_OCCUP     = {"Sans emploi":0,"Agriculture (à son compte)":6,"Services / Commerce":4,"Personnel professionnel ou cadre":1,"Travail manuel qualifié":7,"Travail manuel non qualifié":8,"Employée de maison":5,"Secrétariat / Bureau":3,"Ne sait pas":98}
MAP_CPN       = {"Aucune visite":"0","1 à 3 visites":"1-3","4 visites ou plus (recommandé par l'OMS)":"4+"}
MAP_PARITE    = {"1 à 2 enfants":"1-2","3 à 4 enfants":"3-4","5 enfants ou plus":"5+"}
MAP_FREQ      = {"Jamais":0,"Moins d'une fois par semaine":1,"Au moins une fois par semaine":2}
MAP_DEC       = {"La femme seule":1,"La femme et son mari ensemble":2,"Le mari seul":3,"Une autre personne":6}
MAP_RELIGION  = {"Catholique":1,"Protestant":2,"Charismatique / Apostolique / Non-dénominationnel":96,"Kimbanguiste":5,"Musulman":3,"Témoin de Jéhovah":4,"Animiste / Religion traditionnelle":7,"Armée du Salut":6,"Aucune religion":8,"Autre":9}

OPTS_FREQ    = list(MAP_FREQ.keys())
OPTS_DEC     = list(MAP_DEC.keys())
OPTS_REGIONS = list(MAP_REGION.keys())


#  CONSTRUCTION DU VECTEUR D'ENTRÉE
def build_vector(fd: dict, columns: list) -> pd.DataFrame:
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

    vars_cat = ["v013","v024","v025","v106","v130","v190","v717","v701","v157","v158","v159","v743a","v743b","v743d","parite","cpn_cat"]
    df_enc = pd.get_dummies(df, columns=vars_cat, drop_first=True)
    bc = df_enc.select_dtypes("bool").columns
    df_enc[bc] = df_enc[bc].astype(int)
    return df_enc.reindex(columns=columns, fill_value=0)


#  HELPER — ZONE DE RISQUE 
def zone_risque(prob_sba1: float):
    if prob_sba1 >= 70:
        return "Faible risque",   "vert",   "🟢 Forte probabilité d'accouchement avec assistance qualifiée."
    elif prob_sba1 >= 50:
        return "Risque incertain", "orange", "🟡 La prédiction est 'Assistée' mais avec une marge étroite. Surveillance recommandée."
    else:
        return "Risque élevé",    "rouge",  "🔴 Risque élevé d'accouchement sans assistance qualifiée. Intervention prioritaire recommandée."


#  STATE MANAGEMENT
if "page"        not in st.session_state: st.session_state.page = "accueil"
if "history"     not in st.session_state: st.session_state.history = []
if "last_result" not in st.session_state: st.session_state.last_result = None


#  SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
        <div style='font-size:2.5rem;margin-bottom:.4rem'>🤱</div>
        <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:800;color:#F8FAFC'>SBA Prédicteur</div>
        <div style='font-size:.72rem;color:#64748B;margin-top:.2rem'>RDC · EDS 2023-2024</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    pages = [("","Accueil","accueil"),("","Formulaire","formulaire"),("","Résultat","resultat"),("","Historique","historique")]
    for icon, label, key in pages:
        disabled = (key == "resultat" and st.session_state.last_result is None)
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     disabled=disabled, use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.sidebar.empty() # Évite les bugs de rafraîchissement cyclique sur le cloud
            st.rerun()

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
        st.error("⚠️ Modèle ou colonnes `.pkl` introuvables à la racine du dépôt GitHub.")


#  PAGE 1 : ACCUEIL
if st.session_state.page == "accueil":
    st.markdown("""
    <div class='hero-card'>
        <div style='font-size:3.5rem;margin-bottom:1rem'>🤱</div>
        <h1 style='font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#38BDF8,#818CF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.6rem'>SBA Prédicteur: L'Intelligence Artificielle au service de la santé maternelle en RDC.</h1>
        <p style='font-size:1rem;color:#94A3B8;max-width:600px;margin:0 auto 1.5rem;line-height:1.7'>
            Outil de prédiction du recours à un <strong style='color:#38BDF8'>personnel qualifié lors de l'accouchement</strong> chez les femmes mariées en République Démocratique du Congo, basé sur les données EDS-RDC 2023-2024.
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
        La **mortalité maternelle** en République Démocratique du Congo reste parmi les plus élevées au monde. Une proportion significative des femmes accouchent encore **sans être accompagnées par un professionnel de santé qualifié**: médecin, infirmière ou sage-femme.

        Cette application utilise un **modèle de régression logistique** entraîné sur 10 362 femmes mariées pour prédire si, selon son profil socio-démographique, une femme est susceptible d'accoucher avec ou sans assistance qualifiée.

        Le résultat s'affiche avec une **zone de risque colorée** (🟢 / 🟡 / 🔴) et les probabilités exactes pour chaque classe, afin d'aider les décideurs en santé publique à cibler les interventions.
        """)
    with col_ctx2:
        for val, label, color in [("84.1 %","taux national d'assistées (EDS-RDC)","#34D399"), ("15.9 %","femmes sans assistance qualifiée","#F87171"), ("26 provinces","couverture nationale de l'analyse","#FBBF24")]:
            st.markdown(f"""
            <div class='stat-card'>
                <div style='font-size:2.2rem;font-weight:800;color:{color};font-family:Space Grotesk,sans-serif'>{val}</div>
                <div style='font-size:.8rem;color:#64748B;margin-top:.3rem'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    if st.button("Commencer à remplir le formulaire", use_container_width=True):
        st.session_state.page = "formulaire"
        st.rerun()


#  PAGE 2 : FORMULAIRE
elif st.session_state.page == "formulaire":
    st.markdown("#  Formulaire de prédiction")
    
    if not MODEL_OK:
        st.error("Impossible de lancer la prédiction : assurez-vous que `model.pkl` et `columns.pkl` sont téléversés sur votre GitHub.")
        st.stop()

    with st.form("sba_form"):
        st.markdown("###  Profil démographique")
        g1c1, g1c2, g1c3 = st.columns(3)
        with g1c1: age = st.selectbox("Groupe d'âge *", list(MAP_AGE.keys()))
        with g1c2: residence = st.selectbox("Milieu de résidence *", ["Milieu urbain","Milieu rural"])
        with g1c3: region = st.selectbox("Province de résidence *", OPTS_REGIONS)

        st.markdown("---")
        st.markdown("###  Éducation et situation économique")
        g2c1, g2c2 = st.columns(2)
        with g2c1:
            education = st.selectbox("Niveau d'instruction de la femme *", list(MAP_EDUC.keys()))
            richesse = st.selectbox("Niveau économique du ménage *", list(MAP_RICHESSE.keys()))
        with g2c2:
            edu_mari = st.selectbox("Niveau d'instruction du mari *", list(MAP_EDUC_MARI.keys()))
            occupation = st.selectbox("Activité professionnelle de la femme *", list(MAP_OCCUP.keys()))

        st.markdown("---")
        st.markdown("###  Santé maternelle")
        g3c1, g3c2 = st.columns(2)
        with g3c1: cpn = st.selectbox("Consultations prénatales (CPN) *", list(MAP_CPN.keys()))
        with g3c2: parite = st.selectbox("Nombre total d'enfants (parité) *", list(MAP_PARITE.keys()))

        st.markdown("---")
        st.markdown("### Accès à l'information et aux médias")
        g4c1, g4c2, g4c3 = st.columns(3)
        with g4c1: journal = st.selectbox("Lecture de journaux *", OPTS_FREQ)
        with g4c2: radio = st.selectbox("Écoute de la radio *", OPTS_FREQ)
        with g4c3: tv = st.selectbox("Visionnage de la télévision *", OPTS_FREQ)

        st.markdown("---")
        st.markdown("### Autonomie décisionnelle")
        g5c1, g5c2, g5c3 = st.columns(3)
        with g5c1: dec_sante = st.selectbox("Qui décide des soins de santé ? *", OPTS_DEC)
        with g5c2: dec_achat = st.selectbox("Qui décide des grands achats ? *", OPTS_DEC)
        with g5c3: dec_visite = st.selectbox("Qui décide des visites ? *", OPTS_DEC)

        st.markdown("---")
        religion = st.selectbox("Appartenance religieuse *", list(MAP_RELIGION.keys()))

        submitted = st.form_submit_button("Lancer la prédiction", use_container_width=True, type="primary")

    if submitted:
        fd = {"age":age,"residence":residence,"region":region,"education":education,"edu_mari":edu_mari,"richesse":richesse,"occupation":occupation,"cpn":cpn,"parite":parite,"journal":journal,"radio":radio,"tv":tv,"dec_sante":dec_sante,"dec_achat":dec_achat,"dec_visite":dec_visite,"religion":religion}
        
        X_in = build_vector(fd, MODEL_COLS)
        proba = model.predict_proba(X_in)[0]
        p1 = round(float(proba[1]) * 100, 1)
        p0 = round(float(proba[0]) * 100, 1)
        cls = 1 if float(proba[1]) >= THRESHOLD else 0

        result = {**fd, "classe":cls, "prob_assistee":p1, "prob_non_assistee":p0, "threshold":THRESHOLD, "date":datetime.now().strftime("%d/%m/%Y à %H:%M")}
        st.session_state.last_result = result
        st.session_state.history.append(result)
        st.session_state.page = "resultat"
        st.rerun()


#  PAGE 3 : RÉSULTAT
elif st.session_state.page == "resultat":
    r = st.session_state.last_result
    if r is None:
        st.warning("Aucun résultat disponible.")
        st.stop()

    if r["classe"] == 1:
        st.markdown("<div class='result-main-yes'><h2>✅ Accouchement Assisté prédit</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-main-no'><h2>⚠️ Accouchement Non Assisté prédit</h2></div>", unsafe_allow_html=True)

    niveau, couleur, message = zone_risque(r["prob_assistee"])
    st.markdown(f"<div class='risk-box-{couleur}'><h4>{niveau}</h4><p>{message}</p></div>", unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)
    with pc1:
        st.metric(label="Personnel qualifié (SBA = 1)", value=f"{r['prob_assistee']} %")
        st.progress(r["prob_assistee"] / 100)
    with pc2:
        st.metric(label="Sans assistance (SBA = 0)", value=f"{r['prob_non_assistee']} %")
        st.progress(r["prob_non_assistee"] / 100)

    st.divider()
    if st.button("Nouvelle prédiction", use_container_width=True):
        st.session_state.page = "formulaire"
        st.rerun()


#  PAGE 4 : HISTORIQUE
elif st.session_state.page == "historique":
    st.markdown("# Historique des prédictions")
    if not st.session_state.history:
        st.info("Aucune prédiction enregistrée dans cette session.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        # Calcul de statistiques simples pour finaliser proprement le code initial
        moy_p = round(df_hist["prob_assistee"].mean(), 1)
        st.metric(label="Moyenne des probabilités d'assistance", value=f"{moy_p} %")
