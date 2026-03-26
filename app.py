import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Aramisauto – Analyse Clients",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# PALETTE & STYLES
# ─────────────────────────────────────────
PRIMARY  = "#1B4FD8"
ACCENT   = "#3B82F6"
SUCCESS  = "#16A34A"
WARNING  = "#F59E0B"
NEUTRAL  = "#64748B"
BG_PAGE  = "#F8FAFC"
BG_CARD  = "#FFFFFF"
BORDER   = "#E2E8F0"

CHART_COLORS = [PRIMARY, "#60A5FA", "#34D399", "#FBBF24", "#F87171", "#A78BFA", "#FB923C", "#38BDF8"]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {BG_PAGE};
        color: #1E293B;
    }}
    .block-container {{ padding-top: 3rem; padding-bottom: 2rem; max-width: 1280px; }}

    .page-header {{
        display: flex; align-items: center; gap: 16px;
        padding: 20px 0 18px;
        border-bottom: 2px solid {BORDER};
        margin-bottom: 24px;
    }}
    .page-header-icon {{ font-size: 2rem; line-height: 1; }}
    .page-header h1 {{
        font-size: 1.6rem; font-weight: 700;
        color: #0F172A; margin: 0; letter-spacing: -0.3px;
    }}
    .page-header p {{ font-size: 0.85rem; color: {NEUTRAL}; margin: 2px 0 0; }}

    .kpi-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 18px;
    }}
    .kpi-value {{
        font-size: 1.75rem; font-weight: 700;
        color: {PRIMARY}; margin: 0; line-height: 1.1;
    }}
    .kpi-label {{
        font-size: 0.75rem; color: {NEUTRAL};
        text-transform: uppercase; letter-spacing: 0.6px;
        margin-top: 5px;
    }}

    .section-title {{
        font-size: 0.95rem; font-weight: 600; color: #0F172A;
        padding-bottom: 6px; border-bottom: 1px solid {BORDER};
        margin: 20px 0 12px;
    }}

    .insight-box {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid {PRIMARY};
        border-radius: 8px;
        padding: 14px 18px; margin: 10px 0 20px;
        font-size: 0.875rem; color: #1E3A5F; line-height: 1.6;
    }}
    .insight-title {{
        font-weight: 600; font-size: 0.78rem;
        text-transform: uppercase; letter-spacing: 0.5px;
        color: {PRIMARY}; margin-bottom: 6px;
    }}
    .warn-box {{
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid {WARNING};
        border-radius: 8px;
        padding: 14px 18px; margin: 10px 0 20px;
        font-size: 0.875rem; color: #78350F; line-height: 1.6;
    }}

    .persona-card {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 18px 20px; height: 100%;
    }}
    .persona-badge {{
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 10px;
    }}
    .persona-name {{ font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 6px; }}
    .persona-stat {{ font-size: 0.82rem; color: {NEUTRAL}; margin: 3px 0; }}
    .persona-stat b {{ color: #0F172A; }}

    .reco-card {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
        display: flex; align-items: flex-start; gap: 14px;
    }}
    .reco-rank {{ font-size: 1.3rem; font-weight: 700; color: {PRIMARY}; min-width: 32px; padding-top: 2px; }}
    .reco-body {{ flex: 1; }}
    .reco-model {{ font-size: 0.98rem; font-weight: 600; color: #0F172A; }}
    .reco-detail {{ font-size: 0.8rem; color: {NEUTRAL}; margin-top: 4px; }}
    .reco-detail b {{ color: #0F172A; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: transparent;
        border-bottom: 1px solid {BORDER}; padding-bottom: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px 6px 0 0; font-size: 0.85rem; font-weight: 500;
        padding: 8px 16px; border: 1px solid transparent;
        border-bottom: none; color: {NEUTRAL};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BG_CARD} !important;
        border-color: {BORDER} !important;
        color: {PRIMARY} !important; font-weight: 600 !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background: {BG_CARD}; border: 1px solid {BORDER};
        border-top: none; border-radius: 0 0 10px 10px;
        padding: 20px 24px;
    }}

    [data-testid="stSidebar"] {{
        background-color: {BG_CARD}; border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
        font-size: 0.9rem; font-weight: 600; color: #0F172A;
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def insight(text, title="💡 Interprétation"):
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">{title}</div>
        {text}
    </div>""", unsafe_allow_html=True)

def warn_box(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def chart_defaults(fig, h=320):
    fig.update_layout(
        height=h, margin=dict(t=16, b=10, l=8, r=8),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=12, color="#1E293B"),
        legend=dict(font=dict(size=11)),
        xaxis=dict(showgrid=False, linecolor=BORDER, linewidth=1),
        yaxis=dict(gridcolor="#F1F5F9", linecolor=BORDER),
    )
    return fig


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/df_master.csv")
    df["DATE_PANIER"]    = pd.to_datetime(df["DATE_PANIER"],    errors="coerce")
    df["DATE_CREATION"]  = pd.to_datetime(df["DATE_CREATION"],  errors="coerce")
    df["VEHICULE_MEC"]   = pd.to_datetime(df["VEHICULE_MEC"],   errors="coerce")
    df["AGE"]                      = pd.to_numeric(df["AGE"],                      errors="coerce")
    df["PRIX_VENTE_TTC_COM"]       = pd.to_numeric(df["PRIX_VENTE_TTC_COM"],       errors="coerce")
    df["VEHICULE_KM"]              = pd.to_numeric(df["VEHICULE_KM"],              errors="coerce")
    df["NEAREST_AGENCY_DISTANCE"]  = pd.to_numeric(df["NEAREST_AGENCY_DISTANCE"],  errors="coerce")
    df["GENDER_LABEL"]  = df["GENDER"].map({"H": "Homme", "F": "Femme"})
    df["VEHICULE_AGE"]  = pd.Timestamp.now().year - df["VEHICULE_MEC"].dt.year
    df["TRANCHE_AGE"]   = pd.cut(df["AGE"],
                                  bins=[17, 25, 35, 45, 55, 65, 100],
                                  labels=["18–25","26–35","36–45","46–55","56–65","65+"])
    df["TRANCHE_PRIX"]  = pd.cut(df["PRIX_VENTE_TTC_COM"],
                                  bins=[0, 10000, 15000, 20000, 25000, 30000, 100000],
                                  labels=["<10k","10-15k","15-20k","20-25k","25-30k",">30k"])
    return df

df = load_data()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

# Valeurs par défaut
DEFAULTS = {
    "gender_filter":   ["Homme", "Femme"],
    "age_range":       (18, 90),
    "prix_range":      (int(df["PRIX_VENTE_TTC_COM"].min()), int(df["PRIX_VENTE_TTC_COM"].max())),
    "energie_filter":  sorted(df["VEHICULE_ENERGIE"].unique().tolist()),
    "type_vehicule":   ["VN", "VO"],
    "livraison_filter": df["TYPE_LIVRAISON_COM"].unique().tolist(),
}

# Initialisation session_state
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.markdown("### Filtres")
    st.caption("Appliqués aux onglets Analyse & Clustering")
    st.markdown("")

    gender_filter = st.multiselect("Genre", ["Homme", "Femme"], key="gender_filter")
    age_range = st.slider("Âge", 18, 90, key="age_range")
    prix_range = st.slider("Prix (€)", int(df["PRIX_VENTE_TTC_COM"].min()), int(df["PRIX_VENTE_TTC_COM"].max()), key="prix_range")
    energie_filter = st.multiselect("Énergie", sorted(df["VEHICULE_ENERGIE"].unique()), key="energie_filter")
    type_vehicule = st.multiselect("Type véhicule", ["VN", "VO"], key="type_vehicule")
    livraison_filter = st.multiselect("Type de livraison", df["TYPE_LIVRAISON_COM"].unique().tolist(), key="livraison_filter")

    st.markdown("")
    if st.button("🔄 Réinitialiser les filtres", use_container_width=True):
        for key in DEFAULTS.keys():
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.caption(f"📁 df_master.csv  ·  **{len(df):,}** lignes")

mask = (
    df["GENDER_LABEL"].isin(st.session_state["gender_filter"]) &
    df["AGE"].between(st.session_state["age_range"][0], st.session_state["age_range"][1]) &
    df["PRIX_VENTE_TTC_COM"].between(st.session_state["prix_range"][0], st.session_state["prix_range"][1]) &
    df["VEHICULE_ENERGIE"].isin(st.session_state["energie_filter"]) &
    df["VEHICULE_TYPE"].isin(st.session_state["type_vehicule"]) &
    df["TYPE_LIVRAISON_COM"].isin(st.session_state["livraison_filter"])
)
dff = df[mask].copy()


# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Présentation",
    "📊  Analyse Exploratoire",
    "🔵  Clustering Clients",
    "🎯  Recommandation",
])


# ════════════════════════════════════════
# TAB 1 — PRÉSENTATION
# ════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="page-header">
        <div class="page-header-icon">🚗</div>
        <div>
            <h1>Aramisauto — Analyse des Profils Clients</h1>
            <p>Dashboard marketing · Segmentation · Recommandation de véhicules</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # Calcul des KPI
    nb_marques = df['VEHICULE_MARQUE'].nunique()
    pct_suv = (df["VEHICULE_CATEGORIE"] == "4x4 et SUV").mean() * 100
    pct_hommes = df['GENDER_LABEL'].value_counts(normalize=True).get('Homme', 0) * 100
    prix_moyen = int(df['PRIX_VENTE_TTC_COM'].mean())
    age_moyen = int(df['AGE'].mean())

    # Liste des KPI à afficher
    kpis = [
        (f"{len(df):,}", "Transactions"),
        (f"{nb_marques}", "Marques"),
        (f"{pct_suv:.0f}%", "Part SUV"),
        (f"{prix_moyen:,} €", "Prix moyen"),
        (f"{age_moyen} ans", "Âge moyen"),
        (f"{pct_hommes:.0f}%", "Clients hommes"),
    ]

    # Création des colonnes pour les KPI
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    for col, (val, label) in zip([k1, k2, k3, k4, k5, k6], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    section("🎯 Objectifs marketing")

    st.markdown("""
    Ce dashboard permet de :

    - **Comprendre les clients** : profils, comportements d'achat, préférences véhicules  
    - **Identifier des segments à forte valeur** pour optimiser le ciblage  
    - **Personnaliser les campagnes marketing** (offres, messages, canaux)  
    - **Optimiser les investissements marketing** en se concentrant sur les segments rentables  
    - **Adapter la stratégie produit** selon la demande (SUV, hybride, etc.)
    """)
    
    st.info("👉 Utilisez les onglets ci-dessus pour analyser les comportements, explorer les segments clients et activer des recommandations personnalisées.")

    
    section("📌 Synthèse des insights clés")
    insight("""
    - Les **SUV représentent +50 % des ventes** → segment prioritaire
    - La clientèle est **majoritairement masculine (66 %)** → potentiel féminin sous-exploité
    - Les **36–55 ans = cœur de cible** → campagnes à optimiser sur ce segment
    - L’**électrique reste faible (~4 %)** → opportunité de croissance stratégique
    """, title="📊 Synthèse rapide pour décision marketing")


    section("❓ Questions auxquelles répond ce dashboard")
    st.markdown("""
    - Qui sont les clients d’Aramisauto ?
    - Quels segments génèrent le plus de valeur ?
    - Quels véhicules sont les plus demandés ?
    - Quels profils cibler pour augmenter les ventes ?
    - Comment personnaliser les campagnes marketing ?
    """)

    section("📊 Données utilisées")
    with st.expander("🔍 Aperçu des données brutes"):
        st.dataframe(df.head(20), use_container_width=True)


# ════════════════════════════════════════
# TAB 2 — ANALYSE EXPLORATOIRE
# ════════════════════════════════════════
with tab2:
    st.caption(f"**{len(dff):,} transactions** correspondent aux filtres actifs · Modifiez-les dans la colonne de gauche.")

    sub_client, sub_vehicule = st.tabs(["👤  Profil Clients", "🚘  Profil Véhicules"])

    # ── CLIENTS ──
    with sub_client:
        c1, c2 = st.columns(2)
        with c1:
            section("Achats par tranche d'âge et par genre")

            age_bins = [18, 25, 35, 45, 55, 65, 90]
            age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            dff['AGE_TRANCHE'] = pd.cut(dff['AGE'], bins=age_bins, right=False, labels=age_labels)

            fig = px.histogram(
                dff,
                x="AGE_TRANCHE",
                color="GENDER_LABEL",
                barmode="group",
                color_discrete_map={"Homme": PRIMARY, "Femme": "#F87171"},
                labels={"AGE_TRANCHE": "Âge", "GENDER_LABEL": "Genre"},
                category_orders={"AGE_TRANCHE": age_labels},
            )

            fig.update_layout(
                xaxis_title="Tranche d'âge",
                yaxis_title="Nombre transactions",
                legend_title="Genre",
                bargap=0.2
            )
            fig.update_traces(
                hovertemplate=(
                    "Tranche : <b>%{x}</b><br>"
                    "Nb transactions : <b>%{y}</b><br>"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(chart_defaults(fig, h=320), use_container_width=True)

            insight("La clientèle est majoritairement <strong>masculine (66 %)</strong>. Les tranches <strong>36–45 ans</strong> et <strong>46–55 ans</strong> concentrent le plus de transactions. Les <strong>65+</strong> représentent un segment fidèle à ne pas négliger dans les campagnes de fidélisation.")

        with c2:
            section("Répartition par genre")
            g_counts = dff["GENDER_LABEL"].value_counts().reset_index()
            g_counts.columns = ["Genre", "Nb"]
            g_counts["Pct"] = (g_counts["Nb"] / g_counts["Nb"].sum() * 100).round(1)
            g_counts["Pct_label"] = g_counts["Pct"].astype(str) + " %"

            fig = px.bar(g_counts, x="Genre", y="Pct",
                color="Genre",
                color_discrete_map={"Homme": PRIMARY, "Femme": "#F87171"},
                text="Pct_label",
                labels={"Pct": "Part des clients (%)", "Genre": "Genre"},
                custom_data=["Nb"])

            fig.update_traces(
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=14, color="white", family="Inter"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Nombre de clients : <b>%{customdata[0]:,}</b><br>"
                    "Part du total : <b>%{y} %</b>"
                    "<extra></extra>"
                )
            )

            fig.update_layout(
                showlegend=True,
                legend=dict(title="Genre", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(chart_defaults(fig, h=320), use_container_width=True)
            insight("Le ratio <strong>Homme / Femme est de 66 % / 34 %</strong>. Fort potentiel d'acquisition sur la cible féminine, encore sous-représentée dans le portefeuille client.")

        c3, c4 = st.columns(2)
        with c3:
            section("Type de financement par tranche d'âge")
            tranche_order = ["18–25","26–35","36–45","46–55","56–65","65+"]
            fi_age = dff.copy()

            fi_age["FI_TYPE"] = fi_age["FI_TYPE"].fillna("Cash") # ON EST SUR DE L'INFO ???
            fi_age = fi_age.groupby(["TRANCHE_AGE", "FI_TYPE"], observed=True).size().reset_index(name="Nb")
            total_fi = fi_age.groupby("TRANCHE_AGE", observed=True)["Nb"].transform("sum")
            fi_age["Pct"] = (fi_age["Nb"] / total_fi * 100).round(1)

            fig = px.bar(
                fi_age, x="TRANCHE_AGE", y="Pct",
                color="FI_TYPE",
                barmode="stack",
                color_discrete_sequence=[PRIMARY, ACCENT, "#34D399", "#FBBF24"],
                labels={"TRANCHE_AGE": "Tranche d'âge", "Pct": "Part des clients (%)", "FI_TYPE": "Financement"},
                category_orders={"TRANCHE_AGE": tranche_order},
                custom_data=["Nb", "FI_TYPE"],
            )

            fig.update_traces(
                hovertemplate="<b>%{customdata[1]}</b><br>Part des clients : <b>%{y:.1f}%</b><br>Nb : <b>%{customdata[0]:,}</b><extra></extra>"
            )

            fig.update_layout(yaxis_title="Part des clients (%)", legend_title="Financement")

            st.plotly_chart(chart_defaults(fig), use_container_width=True)

            insight("Le <strong>Cash</strong> est le mode de règlement dominant toutes tranches confondues. Le <strong>Crédit</strong> et la <strong>LOA</strong> sont plus présents chez les 26–55 ans. Note : 61 % des transactions n'ont pas de financement renseigné.")
        
        c5, c6 = st.columns(2)
        with c5:
            section("Mode de livraison")
            livr = dff["TYPE_LIVRAISON_COM"].value_counts().reset_index()
            livr.columns = ["Mode", "Nb"]

            livr["Pct"] = (livr["Nb"] / livr["Nb"].sum() * 100).round(1)
            livr["Pct_label"] = livr["Pct"].astype(str) + " %"

            fig = px.bar(livr, x="Mode", y="Pct",
                         color="Mode",
                         color_discrete_map={"En Agence": PRIMARY, "A Domicile": "#34D399"},
                         text="Pct_label",
                         labels={"Pct": "Part des clients (%)", "Mode": "Mode"},
                custom_data=["Nb"])
            
            fig.update_traces(
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=14, color="white", family="Inter"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Nombre de clients : <b>%{customdata[0]:,}</b><br>"
                    "Part du total : <b>%{y} %</b>"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(chart_defaults(fig, h=320), use_container_width=True)
            insight("<strong>78% des clients</strong> choisissent le retrait en agence. La livraison à domicile (22 %) est un canal en croissance à valoriser dans les communications marketing.")

        with c6:
            section("Type de véhicule par tranche d'âge")
            tranche_order = ["18–25","26–35","36–45","46–55","56–65","65+"]
            veh_age = dff.copy()
            veh_age["TYPE_LABEL"] = veh_age["VEHICULE_TYPE"].map({"VN": "Véhicule neuf", "VO": "Véhicule d'occasion"})
            veh_age = veh_age.groupby(["TRANCHE_AGE", "TYPE_LABEL"], observed=True).size().reset_index(name="Nb")
            total_age = veh_age.groupby("TRANCHE_AGE", observed=True)["Nb"].transform("sum")
            veh_age["Pct"] = (veh_age["Nb"] / total_age * 100).round(1)
            fig = px.bar(
                veh_age, x="TRANCHE_AGE", y="Pct",
                color="TYPE_LABEL",
                barmode="stack",
                color_discrete_map={"Véhicule neuf": PRIMARY, "Véhicule d'occasion": "#34D399"},
                labels={"TRANCHE_AGE": "Tranche d'âge", "Pct": "Part (%)", "TYPE_LABEL": "Type"},
                category_orders={"TRANCHE_AGE": tranche_order},
                custom_data=["Nb", "TYPE_LABEL"],
            )
            fig.update_traces(
                hovertemplate="<b>%{customdata[1]}</b><br>Part des clients : <b>%{y:.1f}%</b><br>Nb de clients : <b>%{customdata[0]:,}</b><extra></extra>"
            )
            fig.update_layout(yaxis_title="Part des clients (%)", legend_title="Type")
            st.plotly_chart(chart_defaults(fig, h=320), use_container_width=True)
            insight("Les clients <strong>jeunes (18–35 ans)</strong> s'orientent davantage vers l'<strong>occasion</strong>, tandis que les tranches 46+ achètent proportionnellement plus de véhicules <strong>neufs</strong>.")

        section("Top 20 villes")
        dff_villes = dff[~dff['VILLE'].isin([np.nan, "x", "X", "NON REPONSE"])]
        top_villes = dff_villes["VILLE"].value_counts().head(20).reset_index()

        top_villes.columns = ["Ville", "Nb"]

        total = top_villes["Nb"].sum()
        top_villes["Pct"] = top_villes["Nb"] / total * 100
        top_villes["Pct_label"] = top_villes["Pct"].map(lambda x: f"{x:.1f}%")

        fig = px.bar(top_villes, x="Pct", y="Ville", orientation="h",
                     color="Pct", color_continuous_scale=["#DBEAFE", PRIMARY],
                     labels={"Pct": "Part de transactions (%)", "Ville": "Ville"})
        fig.update_traces(
                hovertemplate=(
                    "Part de transactions (%) : <b>%{x}</b><br>"
                    "Ville : <b>%{y}</b>"
                    "<extra></extra>"
                )
            )
        
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(chart_defaults(fig, h=500), use_container_width=True)
        insight("Paris domine largement, suivie de Nice, Bordeaux et Strasbourg. Les grandes et moyennes métropoles régionales constituent le cœur de la clientèle — à cibler en priorité dans les campagnes géolocalisées.")

    # ── VÉHICULES ──
    with sub_vehicule:
        # ── Ligne 1 : Équipements | Carburant ──
        e1, e2 = st.columns(2)
        with e1:
            section("Top 10 équipements les plus demandés")
            def parse_equip(val):
                try:
                    return [item["category"] for item in json.loads(val)]
                except Exception:
                    return []
            equip_series = dff["OFFER_EQUIPMENTS_MAIN_LIST"].dropna().apply(parse_equip)
            equip_df = equip_series.explode().value_counts().head(10).reset_index()
            equip_df.columns = ["Équipement", "Nb"]
            fig = px.bar(equip_df, x="Nb", y="Équipement", orientation="h",
                         color="Nb", color_continuous_scale=["#DBEAFE", PRIMARY],
                         labels={"Nb": "Nombre de véhicules", "Équipement": "Équipement"})
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(chart_defaults(fig, h=380), use_container_width=True)
            insight("La <strong>Climatisation</strong> et le <strong>Bluetooth</strong> sont les équipements les plus fréquents. Les équipements connectés (Apple CarPlay, Android Auto) gagnent en popularité — un argument marketing fort pour les cibles jeunes.")

        with e2:
            section("Carburant le plus demandé")
            energie = dff["VEHICULE_ENERGIE"].value_counts().reset_index()
            energie.columns = ["Carburant", "Nb"]
            fig = px.bar(energie, x="Carburant", y="Nb",
                         color="Carburant",
                         color_discrete_sequence=CHART_COLORS,
                         labels={"Nb": "Transactions", "Carburant": "Carburant"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_defaults(fig, h=380), use_container_width=True)
            insight("L'<strong>Essence</strong> domine largement (54 %), suivie de l'<strong>Hybride</strong> (19 %) et du <strong>Diesel</strong> (18 %). La part électrique reste faible (4 %) mais en croissance — à intégrer dans les communications.")

        # ── Ligne 2 : Top 15 marques | Catégories ──
        v1, v2 = st.columns(2)
        with v1:
            section("Top 15 marques vendues")
            marques = dff["VEHICULE_MARQUE"].value_counts().head(15).reset_index()
            marques.columns = ["Marque", "Nb"]
            fig = px.bar(marques, x="Nb", y="Marque", orientation="h",
                         color="Nb", color_continuous_scale=["#DBEAFE", PRIMARY],
                         labels={"Nb": "Transactions"})
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(chart_defaults(fig, h=420), use_container_width=True)
            insight("<strong>Peugeot, Citroën et Renault</strong> trustent le podium et représentent +63 % des ventes. Le catalogue est fortement ancré sur les marques françaises — un levier de partenariat commercial à exploiter.")

        with v2:
            section("Catégories de véhicules")
            cats = dff["VEHICULE_CATEGORIE"].value_counts().reset_index()
            cats.columns = ["Catégorie", "Nb"]
            fig = px.bar(cats, x="Catégorie", y="Nb", color="Catégorie",
                         color_discrete_sequence=CHART_COLORS)
            fig.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(chart_defaults(fig, h=420), use_container_width=True)
            insight("Les <strong>SUV & 4x4</strong> représentent 54 % des ventes. Les Citadines arrivent 2ème (24 %), prisées par les profils urbains. Ces deux catégories devraient concentrer l'essentiel des investissements marketing.")

        section("Prix moyen par marque (Top 15)")
        prix_marque = dff.groupby("VEHICULE_MARQUE")["PRIX_VENTE_TTC_COM"].mean().nlargest(15).reset_index()
        prix_marque.columns = ["Marque", "Prix moyen"]
        fig = px.bar(prix_marque, x="Prix moyen", y="Marque", orientation="h",
                         color="Prix moyen", color_continuous_scale=["#FEF9C3", WARNING])
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(chart_defaults(fig, h=420), use_container_width=True)
        insight("<strong>Tesla, BMW et Mercedes</strong> affichent les prix moyens les plus élevés. Les marques françaises se positionnent sur le segment médian (18 000–25 000 €), cœur de marché d'Aramisauto.")
        
        # ── Ligne 3 : Distribution prix pleine largeur ──
        section("Distribution des prix — VN vs VO")
        fig = px.histogram(dff, x="PRIX_VENTE_TTC_COM", nbins=32,
                           color="VEHICULE_TYPE",
                           color_discrete_map={"VN": PRIMARY, "VO": "#FBBF24"},
                           barmode="overlay", opacity=0.8,
                           histnorm="percent",
                           labels={"PRIX_VENTE_TTC_COM": "Prix (€)", "VEHICULE_TYPE": "Type"})
        fig.update_layout(yaxis_title="Part (%)")
        st.plotly_chart(chart_defaults(fig), use_container_width=True)
        insight("Les <strong>VN</strong> se concentrent entre 20 000 et 35 000 €. Les <strong>VO</strong> sont plus dispersés avec un pic à 15 000 €. Prix moyen global : <strong>21 724 €</strong>.")

        section("Prix par boîte de vitesse et catégorie de véhicule")
        fig = px.box(dff, x="VEHICULE_BOITE_VITESSE", y="PRIX_VENTE_TTC_COM",
                     color="VEHICULE_CATEGORIE",
                     color_discrete_sequence=CHART_COLORS,
                     labels={"VEHICULE_BOITE_VITESSE": "Boîte", "PRIX_VENTE_TTC_COM": "Prix (€)"},
                     points=False)
        st.plotly_chart(chart_defaults(fig, h=380), use_container_width=True)
        insight("La <strong>boîte automatique</strong> est systématiquement associée à des prix plus élevés, toutes catégories confondues. Les SUV automatiques affichent la médiane la plus haute (~27 000 €).")


# ════════════════════════════════════════
# TAB 3 — CLUSTERING
# ════════════════════════════════════════
with tab3:
    st.markdown("#### Segmentation non supervisée de la clientèle")

    insight("<strong>Qu'est-ce que le clustering K-Means ?</strong><br> " \
    "L'algorithme regroupe automatiquement les clients en <em>k</em> segments homogènes en minimisant la variance à l'intérieur de chaque groupe. " \
    "Les variables utilisées sont : <strong>Âge, Prix d'achat, Kilométrage, Distance à l'agence, Genre, Énergie et Catégorie du véhicule</strong>. " \
    "Toutes les variables sont normalisées pour éviter les biais d'échelle. "
    "<br><br> " \
    "Utilisez le curseur ci-dessous pour explorer différents nombres de segments et trouver la segmentation la plus utile pour le marketing.", 
    title="ℹ️ Comment fonctionne cet onglet ?")

    col_param, _ = st.columns([1, 3])
    with col_param:
        n_clusters = st.slider("Nombre de segments (k)", 2, 8, 4)

    @st.cache_data
    def build_cluster_features(data):
        num_cols = ["AGE", "PRIX_VENTE_TTC_COM", "VEHICULE_KM", "NEAREST_AGENCY_DISTANCE"]
        features = data[num_cols].fillna(data[num_cols].median()).copy()
        le = LabelEncoder()
        features["GENDER_ENC"]    = le.fit_transform(data["GENDER"].fillna("H"))
        features["ENERGIE_ENC"]   = le.fit_transform(data["VEHICULE_ENERGIE"].fillna("Essence"))
        features["CATEGORIE_ENC"] = le.fit_transform(data["VEHICULE_CATEGORIE"].fillna("Autre"))
        return features

    @st.cache_data
    def run_kmeans(data, k):
        features = build_cluster_features(data)
        X = StandardScaler().fit_transform(features)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        coords = PCA(n_components=2).fit_transform(X)
        return labels, sil, coords

    labels, sil_score, pca_coords = run_kmeans(dff, n_clusters)
    dff = dff.copy()
    dff["CLUSTER"]       = labels
    dff["CLUSTER_LABEL"] = "Segment " + dff["CLUSTER"].astype(str)
    dff["PCA_1"]         = pca_coords[:, 0]
    dff["PCA_2"]         = pca_coords[:, 1]

    m1, m2, m3 = st.columns(3)
    sil_interp = "Excellent" if sil_score > 0.5 else ("Correct" if sil_score > 0.25 else "Faible — essayez un autre k")
    with m1: st.metric("Score Silhouette", f"{sil_score:.3f}", delta=sil_interp)
    with m2: st.metric("Segments créés", n_clusters)
    with m3: st.metric("Clients analysés", f"{len(dff):,}")

    warn_box("""
    <strong>Score Silhouette</strong> : compris entre -1 et 1.
    Un score <strong>proche de 1</strong> indique des segments bien séparés et compacts (idéal).
    Un score <strong>proche de 0</strong> signifie que les clusters se chevauchent — changez le nombre de segments.
    Sur ce dataset, un score entre 0.15 et 0.30 est attendu compte tenu de la diversité naturelle des profils clients.
    """)

    # ── ACP ──
    section("Visualisation 2D des segments — Analyse en Composantes Principales (ACP)")
    fig_pca = px.scatter(
        dff, x="PCA_1", y="PCA_2", color="CLUSTER_LABEL",
        opacity=0.55,
        hover_data=["AGE", "PRIX_VENTE_TTC_COM", "VEHICULE_MARQUE", "VEHICULE_MODELE"],
        color_discrete_sequence=CHART_COLORS,
        labels={"PCA_1": "Composante 1", "PCA_2": "Composante 2", "CLUSTER_LABEL": "Segment"}
    )
    fig_pca.update_traces(marker=dict(size=5))
    st.plotly_chart(chart_defaults(fig_pca, h=420), use_container_width=True)

    insight("<strong>Comment lire ce graphique ?</strong><br> " \
    "Chaque point représente un client. L'ACP compresse les 7 variables en 2 dimensions pour la visualisation. " \
    "Des nuages <strong>bien séparés</strong> = segments distincts = campagnes très différenciées possibles. " \
    "Un <strong>chevauchement</strong> = profils clients proches sur certaines dimensions — messages plus nuancés. "
    "<br><br> " \
    "👉 <strong>Conseil marketing</strong> : les segments éloignés les uns des autres méritent des offres et " \
    "des canaux de communication radicalement différents.")

    # ── Tableau profil ──
    section("Profil moyen par segment")
    cluster_profile = dff.groupby("CLUSTER").agg(
        Nb_clients=("AGE", "count"),
        Age_moyen=("AGE", "mean"),
        Prix_moyen=("PRIX_VENTE_TTC_COM", "mean"),
        KM_moyen=("VEHICULE_KM", "mean"),
        Dist_agence=("NEAREST_AGENCY_DISTANCE", "mean"),
    ).round(1).reset_index()

    for col_agg, col_name in [("VEHICULE_ENERGIE", "Énergie dom."),
                               ("VEHICULE_CATEGORIE", "Catégorie dom."),
                               ("VEHICULE_MARQUE", "Marque fav.")]:
        top = dff.groupby("CLUSTER")[col_agg].agg(lambda x: x.mode()[0]).reset_index()
        top.columns = ["CLUSTER", col_name]
        cluster_profile = cluster_profile.merge(top, on="CLUSTER")

    cluster_profile.columns = ["Segment", "Nb clients", "Âge moy.", "Prix moy. (€)",
                                "KM moy.", "Dist. agence (km)",
                                "Énergie dom.", "Catégorie dom.", "Marque fav."]
    st.dataframe(cluster_profile, hide_index=True, use_container_width=True)

    insight("Ce tableau synthétise les caractéristiques <strong>moyennes</strong> de chaque segment. " \
    "Focalisez-vous sur <strong>Prix moyen</strong> et <strong>Catégorie dominante</strong> pour évaluer la valeur commerciale de chaque groupe, " \
    "et sur l'<strong>Âge moyen</strong> pour choisir les canaux de communication appropriés (digital vs. traditionnel).")

    # ── Personas ──
    section("Fiches Persona — Portrait de chaque segment")
    badge_colors = ["#DBEAFE","#DCFCE7","#FEF9C3","#FCE7F3","#EDE9FE","#FFEDD5","#E0F2FE","#F0FDF4"]
    text_colors  = [PRIMARY,   SUCCESS,  WARNING,  "#BE185D","#7C3AED","#EA580C","#0284C7","#15803D"]

    cols_p = st.columns(min(n_clusters, 4))
    for i in range(n_clusters):
        row = cluster_profile.iloc[i]
        c = i % 4
        with cols_p[c % len(cols_p)]:
            bg = badge_colors[i % len(badge_colors)]
            tc = text_colors[i % len(text_colors)]
            pct = round(100 * row["Nb clients"] / len(dff), 1)
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-badge" style="background:{bg}; color:{tc}">Segment {i}</div>
                <div class="persona-name">Groupe {i}</div>
                <div class="persona-stat">👥 <b>{int(row['Nb clients']):,}</b> clients ({pct}%)</div>
                <div class="persona-stat">🎂 Âge moyen : <b>{row['Âge moy.']} ans</b></div>
                <div class="persona-stat">💶 Prix moyen : <b>{int(row['Prix moy. (€)']):,} €</b></div>
                <div class="persona-stat">🛣️ KM moyen : <b>{int(row['KM moy.']):,} km</b></div>
                <div class="persona-stat">🔋 Énergie : <b>{row['Énergie dom.']}</b></div>
                <div class="persona-stat">🚗 Catégorie : <b>{row['Catégorie dom.']}</b></div>
                <div class="persona-stat">⭐ Marque fav. : <b>{row['Marque fav.']}</b></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    insight("Ces fiches permettent à l'équipe marketing de <strong>nommer chaque segment</strong> "
    "(ex : 'Famille SUV Hybride', 'Jeune primo-accédant citadine', 'Senior berline premium'). "
    "Chaque persona doit être associé à un <strong>message publicitaire, un canal et une offre spécifiques</strong> "
    "pour maximiser l'efficacité des campagnes.", 
    title="💡 Usage marketing des personas")

    # ── Radar ──
    section("Radar — Comparaison normalisée des segments")
    raw_profile = dff.groupby("CLUSTER")[["AGE","PRIX_VENTE_TTC_COM","VEHICULE_KM","NEAREST_AGENCY_DISTANCE"]].mean()
    raw_norm    = (raw_profile - raw_profile.min()) / (raw_profile.max() - raw_profile.min() + 1e-9)
    raw_norm.columns = ["Âge", "Prix", "Kilométrage", "Distance agence"]

    fig_radar = go.Figure()
    cats = raw_norm.columns.tolist()
    for i, row in raw_norm.iterrows():
        vals = row.tolist() + [row.iloc[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]], fill="toself",
            name=f"Segment {i}",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            fillcolor=CHART_COLORS[i % len(CHART_COLORS)],
            opacity=0.15,
        ))
    fig_radar.update_layout(
        height=420, paper_bgcolor="white",
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER, tickfont=dict(size=10)),
            angularaxis=dict(gridcolor=BORDER)
        ),
        margin=dict(t=30, b=20),
        font=dict(family="Inter"),
        legend=dict(font=dict(size=11))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    insight("<strong>Comment lire le radar ?</strong><br>" \
    " Chaque axe représente une variable normalisée entre 0 et 1. La forme de chaque segment reflète son profil : "
    "<br>• <strong>Grand et régulier</strong> = profil 'équilibré' sur toutes les dimensions "
    "<br>• <strong>Pointu sur un axe</strong> = caractéristique forte et différenciante"
    "<br><br>" \
    "Exemple : un segment fort sur <strong>Prix</strong> et faible sur <strong>Kilométrage</strong> = acheteurs de VN ou quasi-neufs. " \
    "Fort sur <strong>Kilométrage</strong> et faible sur <strong>Prix</strong> = acheteurs de VO à budget contraint.")

    # ── Barres ──
    ca, cb = st.columns(2)
    with ca:
        section("Énergie par segment")
        ener_cl = dff.groupby(["CLUSTER_LABEL","VEHICULE_ENERGIE"]).size().reset_index(name="Nb")
        fig = px.bar(ener_cl, x="CLUSTER_LABEL", y="Nb", color="VEHICULE_ENERGIE",
                     barmode="stack", color_discrete_sequence=CHART_COLORS,
                     labels={"CLUSTER_LABEL": "Segment", "Nb": "Transactions"})
        st.plotly_chart(chart_defaults(fig), use_container_width=True)

    with cb:
        section("Catégorie par segment")
        cat_cl = dff.groupby(["CLUSTER_LABEL","VEHICULE_CATEGORIE"]).size().reset_index(name="Nb")
        fig = px.bar(cat_cl, x="CLUSTER_LABEL", y="Nb", color="VEHICULE_CATEGORIE",
                     barmode="stack", color_discrete_sequence=CHART_COLORS,
                     labels={"CLUSTER_LABEL": "Segment", "Nb": "Transactions"})
        st.plotly_chart(chart_defaults(fig), use_container_width=True)

    insight("Ces deux graphiques permettent de <strong>qualifier la préférence produit</strong> de chaque segment. " \
    "Un segment dominé par des SUV Hybrides à prix élevé = cible <strong>premium / écologique</strong> → arguments confort + impact carbone. " \
    "Un segment de Citadines Essence à faible kilométrage = <strong>jeunes actifs urbains</strong> sensibles au prix → promotions et financement facilité.", 
    title="💡 Lecture combinée Énergie + Catégorie")

    # ── Elbow ──
    with st.expander("📈 Aide au choix du nombre de segments — Méthode du coude"):
        insight("<strong>Comment choisir le bon k ?</strong><br>" \
        "• <strong>Courbe d'inertie</strong> (bleue) : mesure la compacité des clusters. " \
        "Elle décroît à chaque k ajouté. Cherchez le <em>coude</em> — le point où le gain devient marginal.<br> " \
        "• <strong>Score Silhouette</strong> (rouge) : mesure la séparabilité entre clusters. " \
        "Choisissez le k qui <strong>maximise ce score</strong>.<br><br> " \
        "En règle générale, le k optimal se situe entre 3 et 6 sur des données retail.")

        @st.cache_data
        def compute_inertia(data):
            X = StandardScaler().fit_transform(build_cluster_features(data))
            inertias, sils = [], []
            for k in range(2, 9):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbl = km.fit_predict(X)
                inertias.append(km.inertia_)
                sils.append(silhouette_score(X, lbl))
            return list(range(2, 9)), inertias, sils

        ks, inertias, sils = compute_inertia(dff)
        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(
            go.Scatter(x=ks, y=inertias, name="Inertie", mode="lines+markers",
                       line=dict(color=PRIMARY, width=2), marker=dict(size=7)),
            secondary_y=False)
        fig_elbow.add_trace(
            go.Scatter(x=ks, y=sils, name="Silhouette", mode="lines+markers",
                       line=dict(color="#F87171", width=2, dash="dot"), marker=dict(size=7)),
            secondary_y=True)
        fig_elbow.update_layout(height=340, xaxis_title="k (nombre de segments)",
                                paper_bgcolor="white", plot_bgcolor="white",
                                font=dict(family="Inter"), margin=dict(t=10, b=10))
        fig_elbow.update_yaxes(title_text="Inertie (↓ mieux)", secondary_y=False, gridcolor="#F1F5F9")
        fig_elbow.update_yaxes(title_text="Silhouette (↑ mieux)", secondary_y=True)
        st.plotly_chart(fig_elbow, use_container_width=True)


# ════════════════════════════════════════
# TAB 4 — RECOMMANDATION
# ════════════════════════════════════════
with tab4:
    st.markdown("#### Recommandation de véhicules personnalisée")

    insight("<strong>Comment fonctionne cet outil ?</strong><br> " \
    "Renseignez le profil d'un prospect. L'algorithme <strong>K-Nearest Neighbors (KNN)</strong> " \
    "identifie les 20 transactions historiques les plus proches de ce profil en termes d'âge, de budget, " \
    "de kilométrage et de préférences. Il agrège ensuite les modèles de véhicules les plus fréquemment " \
    "achetés dans ce voisinage et vous retourne une liste triée par pertinence. "
    "<br><br> " \
    "👉 Ces recommandations peuvent alimenter des <strong>emails personnalisés, des relances téléphoniques " \
    "ou des pages produit dynamiques</strong> sur le site Aramisauto.", 
    title="ℹ️ Fonctionnement de la recommandation KNN")

    @st.cache_data
    def build_reco_model(data):
        feats = data[["AGE","PRIX_VENTE_TTC_COM","VEHICULE_KM"]].fillna(0).copy()
        le_g = LabelEncoder(); le_e = LabelEncoder()
        le_c = LabelEncoder(); le_b = LabelEncoder()
        feats["GENDER_ENC"]  = le_g.fit_transform(data["GENDER"].fillna("H"))
        feats["ENERGIE_ENC"] = le_e.fit_transform(data["VEHICULE_ENERGIE"].fillna("Essence"))
        feats["CAT_ENC"]     = le_c.fit_transform(data["VEHICULE_CATEGORIE"].fillna("Autre"))
        feats["BV_ENC"]      = le_b.fit_transform(data["VEHICULE_BOITE_VITESSE"].fillna("Boite manuelle"))
        scaler = StandardScaler()
        X      = scaler.fit_transform(feats)
        knn    = NearestNeighbors(n_neighbors=20, metric="euclidean")
        knn.fit(X)
        return knn, scaler, le_g, le_e, le_c, le_b

    knn_model, reco_scaler, le_g, le_e, le_c, le_b = build_reco_model(df)

    section("Renseignez le profil du prospect")
    col1, col2, col3 = st.columns(3)
    with col1:
        r_age    = st.slider("Âge", 18, 90, 35)
        r_budget = st.slider("Budget maximum (€)", 5000, 60000, 20000, step=500)
    with col2:
        r_gender  = st.selectbox("Genre", ["H","F"], format_func=lambda x: "Homme" if x=="H" else "Femme")
        r_energie = st.selectbox("Énergie préférée", sorted(df["VEHICULE_ENERGIE"].unique().tolist()))
    with col3:
        r_categorie = st.selectbox("Catégorie souhaitée", df["VEHICULE_CATEGORIE"].unique().tolist())
        r_boite     = st.selectbox("Boîte de vitesse", ["Boite manuelle","Boite auto."])
    r_km_max = st.slider("Kilométrage maximum (km)", 0, 160000, 50000, step=5000)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Trouver les véhicules adaptés", type="primary", use_container_width=True):
        with st.spinner("Recherche en cours…"):
            def safe_enc(le, val):
                try: return le.transform([val])[0]
                except: return 0

            g_enc = safe_enc(le_g, r_gender)
            e_enc = safe_enc(le_e, r_energie)
            c_enc = safe_enc(le_c, r_categorie)
            b_enc = safe_enc(le_b, r_boite)

            input_scaled = reco_scaler.transform(
                np.array([[r_age, r_budget, r_km_max / 2, g_enc, e_enc, c_enc, b_enc]])
            )
            distances, indices = knn_model.kneighbors(input_scaled)
            neighbors = df.iloc[indices[0]].copy()
            neighbors = neighbors[
                (neighbors["PRIX_VENTE_TTC_COM"] <= r_budget) &
                (neighbors["VEHICULE_KM"]        <= r_km_max)
            ]

            reco = (neighbors
                    .groupby(["VEHICULE_MARQUE","VEHICULE_MODELE"])
                    .agg(
                        nb=("VEHICULE_MODELE", "count"),
                        prix_moyen=("PRIX_VENTE_TTC_COM", "mean"),
                        km_moyen=("VEHICULE_KM", "mean"),
                        energie=("VEHICULE_ENERGIE",   lambda x: x.mode()[0] if len(x) else ""),
                        categorie=("VEHICULE_CATEGORIE", lambda x: x.mode()[0] if len(x) else ""),
                    )
                    .sort_values("nb", ascending=False)
                    .head(8).reset_index())

            section("Résultats")

            if len(reco) == 0:
                st.warning("Aucun véhicule trouvé. Essayez d'augmenter le budget ou le kilométrage maximum.")
            else:
                energie_icons = {"Essence":"⛽","Diesel":"🛢️","Hybride":"🔋","Electrique":"⚡","Autre":"🔧"}
                col_l, col_r = st.columns(2)
                for i, (_, row) in enumerate(reco.iterrows()):
                    icon = energie_icons.get(row["energie"], "🚗")
                    with (col_l if i % 2 == 0 else col_r):
                        st.markdown(f"""
                        <div class="reco-card">
                            <div class="reco-rank">#{i+1}</div>
                            <div class="reco-body">
                                <div class="reco-model">{icon}&nbsp; {row['VEHICULE_MARQUE']} {row['VEHICULE_MODELE']}</div>
                                <div class="reco-detail">
                                    💶 Prix moyen : <b>{int(row['prix_moyen']):,} €</b>
                                    &nbsp;·&nbsp; 🛣️ KM moyen : <b>{int(row['km_moyen']):,} km</b>
                                    &nbsp;·&nbsp; 📦 <b>{row['categorie']}</b>
                                    &nbsp;·&nbsp; <b>{row['energie']}</b>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)

                insight(f"""Les véhicules listés correspondent à ceux achetés par des clients au profil similaire dans l'historique 
                        (<strong>{r_age} ans, budget {r_budget:,} €, {r_energie} · {r_categorie}</strong>).
                        Le modèle #1 est celui le plus fréquemment acheté parmi les 20 clients les plus proches.
                        <br><br>
                        👉 <strong>Pour l'équipe commerciale</strong> : utilisez ces résultats pour personnaliser
                        les emails de nurturing, les relances ou les landing pages produit.""",
                        title="💡 Comment utiliser ces résultats ?")

                section("Comparatif prix des véhicules recommandés vs votre budget")
                fig_reco = px.bar(
                    reco, x="VEHICULE_MODELE", y="prix_moyen",
                    color="VEHICULE_MARQUE",
                    color_discrete_sequence=CHART_COLORS,
                    text=reco["prix_moyen"].apply(lambda x: f"{int(x):,} €"),
                    labels={"VEHICULE_MODELE": "Modèle", "prix_moyen": "Prix moyen (€)", "VEHICULE_MARQUE": "Marque"}
                )
                fig_reco.update_traces(textposition="outside")
                fig_reco.add_hline(
                    y=r_budget, line_dash="dot", line_color="#EF4444",
                    annotation_text=f"Budget max : {r_budget:,} €",
                    annotation_font=dict(color="#EF4444")
                )
                st.plotly_chart(chart_defaults(fig_reco, h=380), use_container_width=True)
