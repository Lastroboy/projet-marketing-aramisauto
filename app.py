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
import prince
from scipy import stats

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
    st.caption("Appliqués aux onglets Analyse")
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
    "🔬  Analyse & Hypothèses",
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
    st.info("👉 Tous les graphiques sont interactifs : survolez-les pour afficher des informations détaillées.")


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

            insight("La clientèle est majoritairement <strong>masculine (66 %)</strong>. Les tranches <strong>35–44 ans</strong> et <strong>45–54 ans</strong> concentrent beaucoup de transactions. Les <strong>65+</strong> représentent le segment le plus fidèle à ne pas négliger dans les campagnes de fidélisation.")

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
                showlegend=False,
                legend=dict(title="Genre", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(chart_defaults(fig, h=320), use_container_width=True)
            insight("Le ratio <strong>Homme / Femme est de 66 % / 34 %</strong>. Fort potentiel d'acquisition sur la cible féminine, encore sous-représentée dans le portefeuille client.")

        c3, c4 = st.columns(2)
        with c3:
            section("Type de financement par tranche d'âge")
            tranche_order = ["18–25","26–35","36–45","46–55","56–65","65+"]
            fi_age = dff.copy()

            fi_age["FI_TYPE"] = fi_age["FI_TYPE"].fillna("Comptant")
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

            insight("Le <strong>Paiement au comptant</strong> est le mode de règlement dominant toutes tranches confondues. Le <strong>Crédit</strong> et la <strong>LOA</strong> sont plus présents chez les 18-25 ans.")
        
        with c4:
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

        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_defaults(fig, h=320), use_container_width=True)
        insight("<strong>78% des clients</strong> choisissent le retrait en agence. La livraison à domicile (22 %) est un canal en croissance à valoriser dans les communications marketing.")

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
                    "Part de transactions : <b>%{x:.2f}%</b><br>"
                    "Ville : <b>%{y}</b>"
                    "<extra></extra>"
                )
            )
        
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(chart_defaults(fig, h=500), use_container_width=True)
        insight("Paris domine largement, suivie de Nice, Bordeaux et Strasbourg. Les grandes et moyennes métropoles régionales constituent le cœur de la clientèle — à cibler en priorité dans les campagnes géolocalisées.")

    # ── VÉHICULES ──
    with sub_vehicule:
        e1, e2 = st.columns(2)
        with e1:
            # Graphique sur le Top 10 des équipements les plus demandés
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
            
            fig.update_traces(
                hovertemplate="<br>Nb de véhicules : <b>%{x}</b><br>Equipement : <b>%{y}</b><extra></extra>"
                )
            
            fig.update_layout(yaxis_title="Equipements", coloraxis_showscale=False, yaxis=dict(autorange="reversed"))         

            st.plotly_chart(chart_defaults(fig, h=380), use_container_width=True)
            insight("La <strong>Climatisation</strong> et le <strong>Bluetooth</strong> sont les équipements les plus fréquents. Les équipements connectés (Apple CarPlay, Android Auto) sont également demandés — un argument marketing fort pour les cibles jeunes.")

        with e2:
            # Graphique sur les carburants les plus demandés
            section("Carburants les plus demandés")
            energie = dff["VEHICULE_ENERGIE"].value_counts().reset_index()
            energie.columns = ["Carburant", "Nb"]

            energie["Pct"] = (energie["Nb"] / energie["Nb"].sum() * 100).round(1)
            energie["Pct_label"] = energie["Pct"].astype(str) + " %"

            fig = px.bar(energie, x="Carburant", y="Pct",
                         color="Carburant",
                         color_discrete_sequence=CHART_COLORS,
                         labels={"Pct": "Part de véhicules (%)", "Carburant": "Carburant"})
            
            fig.update_traces(
                hovertemplate="<br>Carburant : <b>%{x}</b><br>Part de véhicules : <b>%{y}%</b><extra></extra>"
                )
            
            fig.update_layout(yaxis_title="Part des véhicules (%)", xaxis_title="Carburants", showlegend=False)
            st.plotly_chart(chart_defaults(fig, h=380), use_container_width=True)
            insight("L'<strong>Essence</strong> domine largement (54 %), suivie de l'<strong>Hybride</strong> (19 %) et du <strong>Diesel</strong> (18 %). La part électrique reste faible (4 %) — à intégrer dans les communications.")

        v1, v2 = st.columns(2)
        with v1:
            # Graphique sur le Top 15 des marques les plus vendues
            section("Top 15 marques vendues")
            marques = dff["VEHICULE_MARQUE"].value_counts().head(15).reset_index()
            marques.columns = ["Marque", "Nb"]

            fig = px.bar(marques, x="Nb", y="Marque", orientation="h",
                         color="Nb", color_continuous_scale=["#DBEAFE", PRIMARY],
                         labels={"Nb": "Transactions"})
            
            fig.update_traces(
                hovertemplate="<br>Nb de transactions : <b>%{x}</b><br>Marque : <b>%{y}</b><extra></extra>"
                )
            
            fig.update_layout(yaxis_title="Marques", xaxis_title="Nombre de transactions", coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(chart_defaults(fig, h=420), use_container_width=True)
            insight("<strong>Peugeot, Citroën et Renault</strong> trustent le podium et représentent +63 % des ventes. Le catalogue est fortement ancré sur les marques françaises — un levier de partenariat commercial à exploiter.")

        with v2:
            section("Catégories de véhicules")
            cats = dff["VEHICULE_CATEGORIE"].value_counts().reset_index()
            cats.columns = ["Catégorie", "Nb"]

            cats["Pct"] = (cats["Nb"] / cats["Nb"].sum() * 100).round(1)
            cats["Pct_label"] = cats["Pct"].astype(str) + " %"

            fig = px.bar(cats, x="Catégorie", y="Pct", color="Catégorie",
                         color_discrete_sequence=CHART_COLORS)
            
            fig.update_traces(
                hovertemplate="<br>Catégorie : <b>%{x}</b><br>Part des véhicules : <b>%{y}%</b><extra></extra>"
                )
            
            fig.update_layout(yaxis_title="Part des véhicules (%)", xaxis_title="Catégories", showlegend=False, xaxis_tickangle=-25)

            st.plotly_chart(chart_defaults(fig, h=420), use_container_width=True)
            insight("Les <strong>SUV & 4x4</strong> représentent 54 % des ventes. Les Citadines arrivent 2ème (24 %), prisées par les profils urbains. Ces deux catégories devraient concentrer l'essentiel des investissements marketing.")

    
        # Graphique sur la distribution des prix (VN vs VO)
        section("Distribution des prix — Véhicules Neuf (VN) vs Véhicules d'Occasion (VO)")
        fig = px.histogram(dff, x="PRIX_VENTE_TTC_COM", nbins=32,
                           color="VEHICULE_TYPE",
                           color_discrete_map={"VN": PRIMARY, "VO": "#FBBF24"},
                           barmode="overlay", opacity=0.8,
                           histnorm="percent",
                           labels={"PRIX_VENTE_TTC_COM": "Prix (€)", "VEHICULE_TYPE": "Type"})
        
        
        fig.update_traces(
            hovertemplate="<br>Type : <b>%{fullData.name}</b>"
                        "<br>Prix (€) : <b>%{x}</b>"
                        "<br>Part des transactions : <b>%{y:.2f}%</b>"
                        "<extra></extra>")

        fig.update_layout(yaxis_title="Part des transactions (%)")
        st.plotly_chart(chart_defaults(fig), use_container_width=True)
        insight("Les <strong>véhicule neuf </strong> se concentrent entre 20 000 et 35 000 €. Les <strong>véhicule d'occasion</strong> sont plus dispersés avec un pic à 15 000 €. Prix moyen global : <strong>21 724 €</strong>.")

        section("Prix par boîte de vitesse et catégorie de véhicule")
        fig = px.box(dff, x="VEHICULE_BOITE_VITESSE", y="PRIX_VENTE_TTC_COM",
                     color="VEHICULE_CATEGORIE",
                     color_discrete_sequence=CHART_COLORS,
                     labels={"VEHICULE_BOITE_VITESSE": "Boîte", "PRIX_VENTE_TTC_COM": "Prix (€)"},
                     points=False)
        st.plotly_chart(chart_defaults(fig, h=380), use_container_width=True)
        insight("La <strong>boîte automatique</strong> est systématiquement associée à des prix plus élevés, toutes catégories confondues. Les SUV automatiques affichent la médiane la plus haute (~27 000 €).")


# ════════════════════════════════════════
# TAB 3 — ANALYSE & HYPOTHÈSES
# ════════════════════════════════════════
with tab3:
    st.markdown("#### Analyse exploratoire — Test des hypothèses marketing")

    sub_methodo, sub_h1, sub_h23 = st.tabs([
        "📋  Méthodologie & Hypothèses",
        "H1 — Corrélation de Pearson",
        "H2 & H3 — Chi-deux et ACM",
    ])

    # SOUS-ONGLET 1 — MÉTHODOLOGIE
    with sub_methodo:
        st.markdown("#### Présentation des hypothèses et intuitions")

        insight(
            "Avant de lancer les tests statistiques, nous formulons trois hypothèses marketing "
            "fondées sur des intuitions comportementales. Chaque hypothèse est ensuite vérifiée "
            "de manière rigoureuse dans les sous-onglets suivants.<br><br>"
            "<strong>Méthodes mobilisées</strong> :<br>"
            "• <strong>H1</strong> → Corrélation de Pearson (variable continue × variable continue)<br>"
            "• <strong>H2 & H3</strong> → Test du Chi-deux (variable catégorielle × variable catégorielle) "
            "puis Analyse des Correspondances Multiples (ACM) pour une vision synthétique",
            title="ℹ️ Démarche analytique"
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Carte H1
        st.markdown("""
        <div style="background:#DBEAFE;border-left:5px solid #1B4FD8;border-radius:10px;
                    padding:20px 24px;margin-bottom:18px;">
            <div style="font-size:11px;font-weight:700;color:#1E40AF;letter-spacing:.08em;
                        margin-bottom:6px;">HYPOTHÈSE 1</div>
            <div style="font-size:17px;font-weight:800;color:#1B4FD8;margin-bottom:10px;">
                H1 — Le prix du véhicule acheté augmente-t-il avec l'âge de l'acheteur ?
            </div>
            <div style="font-size:13px;color:#1E3A5F;line-height:1.65;">
                <strong>Intuition :</strong> En avançant dans la vie, un individu accumule du capital —
                grâce à l'évolution de sa carrière, à l'épargne et à la transmission patrimoniale.
                Plus on est âgé, plus on dispose théoriquement des moyens d'acquérir un véhicule
                de gamme supérieure. On s'attend donc à une <strong>corrélation positive</strong>
                entre l'âge et le prix d'achat du véhicule.<br><br>
                <strong>Test utilisé :</strong> Corrélation de Pearson ·
                <strong>Variable X :</strong> Âge ·
                <strong>Variable Y :</strong> Prix de vente TTC
            </div>
        </div>""", unsafe_allow_html=True)

        # Carte H2
        st.markdown("""
        <div style="background:#DCFCE7;border-left:5px solid #16A34A;border-radius:10px;
                    padding:20px 24px;margin-bottom:18px;">
            <div style="font-size:11px;font-weight:700;color:#14532D;letter-spacing:.08em;
                        margin-bottom:6px;">HYPOTHÈSE 2</div>
            <div style="font-size:17px;font-weight:800;color:#15803D;margin-bottom:10px;">
                H2 — Le genre influence-t-il la catégorie de véhicule achetée ?
            </div>
            <div style="font-size:13px;color:#14532D;line-height:1.65;">
                <strong>Intuition :</strong> Des représentations sociales différenciées associent
                souvent les hommes à des véhicules puissants (berlines, sportives) et les femmes
                à des véhicules perçus comme pratiques ou compacts (citadines, SUV familiaux).
                Si ces préférences se retrouvent dans les données, cela justifierait une
                <strong>segmentation créative par genre</strong> selon la catégorie de véhicule.<br><br>
                <strong>Test utilisé :</strong> Chi-deux d'indépendance ·
                <strong>Variable 1 :</strong> Genre ·
                <strong>Variable 2 :</strong> Catégorie de véhicule
            </div>
        </div>""", unsafe_allow_html=True)

        # Carte H3
        st.markdown("""
        <div style="background:#FEF9C3;border-left:5px solid #F59E0B;border-radius:10px;
                    padding:20px 24px;margin-bottom:18px;">
            <div style="font-size:11px;font-weight:700;color:#78350F;letter-spacing:.08em;
                        margin-bottom:6px;">HYPOTHÈSE 3</div>
            <div style="font-size:17px;font-weight:800;color:#B45309;margin-bottom:10px;">
                H3 — L'âge prédit-il le mode de financement choisi ?
            </div>
            <div style="font-size:13px;color:#78350F;line-height:1.65;">
                <strong>Intuition :</strong> Les jeunes acheteurs, dont les revenus sont encore
                en construction et l'épargne limitée, auraient davantage recours à des solutions
                de financement étalées dans le temps — LOA ou crédit. À l'inverse, les clients
                plus âgés, ayant accumulé un patrimoine, disposeraient plus facilement de liquidités
                pour régler comptant. On s'attend donc à un <strong>glissement progressif de la LOA
                vers le paiement comptant</strong> à mesure que l'âge augmente.<br><br>
                <strong>Test utilisé :</strong> Chi-deux d'indépendance ·
                <strong>Variable 1 :</strong> Tranche d'âge ·
                <strong>Variable 2 :</strong> Mode de financement
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        insight(
            "Ces trois hypothèses forment un <strong>fil conducteur cohérent</strong> : "
            "l'âge structure à la fois le budget d'achat (H1) et les préférences de financement (H3), "
            "tandis que le genre oriente le choix de la catégorie de véhicule (H2). "
            "Ensemble, elles permettent de définir des <strong>profils clients actionnables</strong> "
            "pour les équipes marketing et commerciales.",
            title="💡 Cohérence des hypothèses"
        )

    # SOUS-ONGLET 2
    # Hypothèse H1 - Le prix du véhicule augmente-t-il avec l'âge de l'acheteur ?
    with sub_h1:
        section("H1 — Le prix du véhicule augmente-t-il avec l'âge de l'acheteur ?")

        h1_data = dff.dropna(subset=["AGE", "PRIX_VENTE_TTC_COM"]).copy()
        r_val, p_val = stats.pearsonr(h1_data["AGE"], h1_data["PRIX_VENTE_TTC_COM"])

        m, b = np.polyfit(h1_data["AGE"], h1_data["PRIX_VENTE_TTC_COM"], 1)
        x_line = np.linspace(h1_data["AGE"].min(), h1_data["AGE"].max(), 200)
        y_line = m * x_line + b

        fig_h1 = px.scatter(
            h1_data, x="AGE", y="PRIX_VENTE_TTC_COM",
            color="GENDER_LABEL",
            color_discrete_map={"Homme": PRIMARY, "Femme": "#F87171"},
            opacity=0.4,
            labels={"AGE": "Âge", "PRIX_VENTE_TTC_COM": "Prix d'achat (€)", "GENDER_LABEL": "Genre"},
            custom_data=["GENDER_LABEL", "VEHICULE_MARQUE", "VEHICULE_MODELE"]
        )
        fig_h1.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="#0F172A", width=2, dash="dash"),
            name="Régression linéaire", showlegend=True
        ))

        fig_h1.update_traces(marker=dict(size=5), selector=dict(mode="markers"), 
                             hovertemplate="<br>Genre : <b>%{customdata[0]}</b>"
                                        "<br>Âge : <b>%{x}</b>"
                                        "<br>Prix : <b>%{y} €</b>"
                                        "<br>Marque : <b>%{customdata[1]}</b>"
                                        "<br>Modèle : <b>%{customdata[2]}</b>"
                                        "<extra></extra>")
        
        st.plotly_chart(chart_defaults(fig_h1, h=400), use_container_width=True)

        sig_label = "très fortement significatif" if p_val < 0.001 else ("significatif" if p_val < 0.05 else "non significatif")
        confirmed = p_val < 0.05
        box_color = "#D1FAE5" if confirmed else "#FEE2E2"
        border_color = SUCCESS if confirmed else "#EF4444"
        verdict_icon = "✅" if confirmed else "❌"
        
        st.markdown(f"""
        <div style="background:{box_color}; border-left:4px solid {border_color};
                    border-radius:8px; padding:14px 18px; margin:10px 0 16px; font-size:0.875rem;">
            <strong>{verdict_icon} H1 — Résultat</strong><br>
            Corrélation de Pearson : <strong>r = {r_val:.3f}</strong> · p-value : <strong>{p_val:.2e}</strong> ({sig_label})<br>
            {"La corrélation positive est confirmée : les clients plus âgés tendent à acheter des véhicules plus chers."
             if confirmed else
             "La corrélation n'est pas statistiquement significative sur ces données."}
        </div>""", unsafe_allow_html=True)

        insight(
            f"Avec r = {r_val:.3f}, la corrélation est <strong>positive mais modérée</strong>. "
            "L'âge explique une partie de la variance du budget, mais d'autres facteurs jouent aussi (genre, catégorie souhaitée). "
            "<br><br>"
            "👉 <strong>Usage marketing</strong> : les campagnes premium peuvent cibler les 45+ sans exclure les segments plus jeunes à fort budget.",
            title="💡 Interprétation H1"
        )

    # SOUS-ONGLET 3 — H2, H3 & ACM
    with sub_h23:

        # Hypothèse H2 - Le genre influence-t-il la catégorie de véhicule achetée ?
        section("H2 — Le genre influence-t-il la catégorie de véhicule achetée ?")

        h2_data = dff.dropna(subset=["GENDER", "VEHICULE_CATEGORIE"]).copy()
        ct_h2_raw = pd.crosstab(h2_data["GENDER_LABEL"], h2_data["VEHICULE_CATEGORIE"])
        chi2_h2, p_h2, dof_h2, _ = stats.chi2_contingency(ct_h2_raw)

        ct_h2_pct = ct_h2_raw.div(ct_h2_raw.sum(axis=1), axis=0) * 100
        ct_h2_plot = ct_h2_pct.reset_index().melt(id_vars="GENDER_LABEL", var_name="Catégorie", value_name="Pct")

        fig_h2 = px.bar(
            ct_h2_plot, x="GENDER_LABEL", y="Pct", color="Catégorie",
            barmode="stack", color_discrete_sequence=CHART_COLORS,
            labels={"GENDER_LABEL": "Genre", "Pct": "Part (%)"},
            text=ct_h2_plot["Pct"].apply(lambda x: f"{x:.0f}%" if x > 5 else ""),
            custom_data=["Catégorie"]
        )

        fig_h2.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=
                "<br>Catégorie : <b>%{customdata[0]}</b>"
                "<br>Genre : <b>%{x}</b>"
                "<br>Part : <b>%{y:.1f} %</b>"
                "<extra></extra>"
        )
        
        fig_h2.update_layout(yaxis_title="Part (%)", xaxis_title="Genre", legend_title="Catégorie")
        st.plotly_chart(chart_defaults(fig_h2, h=360), use_container_width=True)

        sig_h2 = "très fortement significatif" if p_h2 < 0.001 else ("significatif" if p_h2 < 0.05 else "non significatif")
        conf_h2 = p_h2 < 0.05
        bc2 = "#D1FAE5" if conf_h2 else "#FEE2E2"
        brd2 = SUCCESS if conf_h2 else "#EF4444"
        st.markdown(f"""
        <div style="background:{bc2}; border-left:4px solid {brd2};
                    border-radius:8px; padding:14px 18px; margin:10px 0 16px; font-size:0.875rem;">
            <strong>{"✅" if conf_h2 else "❌"} H2 — Résultat</strong><br>
            Chi-deux : <strong>χ² = {chi2_h2:.1f}</strong> · ddl = {dof_h2} · p-value : <strong>{p_h2:.2e}</strong> ({sig_h2})<br>
            {"Le genre est significativement associé à la catégorie de véhicule achetée."
             if conf_h2 else
             "Aucune association significative détectée entre genre et catégorie."}
        </div>""", unsafe_allow_html=True)

        insight(
            "Le test du Chi-deux mesure si la distribution des catégories est <strong>indépendante du genre</strong>. "
            "Un p < 0.05 signifie que les différences observées ne sont pas dues au hasard. "
            "<br><br>"
            "👉 <strong>Usage marketing</strong> : si H2 est confirmée, des campagnes segmentées par genre sur certaines catégories sont justifiées.",
            title="💡 Interprétation H2"
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Hypothèse H3 -  L'âge prédit-il le mode de financement choisi ?
        section("H3 — L'âge prédit-il le mode de financement choisi ?")

        h3_data = dff.copy()
        h3_data["FI_TYPE"] = h3_data["FI_TYPE"].fillna("Comptant")
        h3_data = h3_data.dropna(subset=["AGE"]).copy()
        h3_data["TRANCHE_AGE"] = pd.cut(
            h3_data["AGE"], bins=[17, 25, 35, 45, 55, 65, 100],
            labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"]
        )
        h3_data = h3_data.dropna(subset=["TRANCHE_AGE"])

        ct_h3_raw  = pd.crosstab(h3_data["TRANCHE_AGE"], h3_data["FI_TYPE"])
        chi2_h3, p_h3, dof_h3, _ = stats.chi2_contingency(ct_h3_raw)

        ct_h3_pct  = ct_h3_raw.div(ct_h3_raw.sum(axis=1), axis=0) * 100
        ct_h3_plot = ct_h3_pct.reset_index().melt(id_vars="TRANCHE_AGE", var_name="Financement", value_name="Pct")
        ct_h3_plot["TRANCHE_AGE"] = ct_h3_plot["TRANCHE_AGE"].astype(str)

        fig_h3 = px.bar(
            ct_h3_plot, x="TRANCHE_AGE", y="Pct", color="Financement",
            barmode="stack", color_discrete_sequence=CHART_COLORS,
            labels={"TRANCHE_AGE": "Tranche d'âge", "Pct": "Part (%)"},
            category_orders={"TRANCHE_AGE": ["18–25","26–35","36–45","46–55","56–65","65+"]},
            text=ct_h3_plot["Pct"].apply(lambda x: f"{x:.0f}%" if x > 6 else ""),
            custom_data=["Financement"]
        )

        fig_h3.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=
                "<br>Financement : <b>%{fullData.name}</b>"
                "<br>Tranche d'âge : <b>%{x}</b>"
                "<br>Part : <b>%{y:.1f} %</b>"
                "<extra></extra>"
        )

        fig_h3.update_layout(yaxis_title="Part (%)", xaxis_title="Tranche d'âge", legend_title="Financement")
        st.plotly_chart(chart_defaults(fig_h3, h=360), use_container_width=True)

        sig_h3 = "très fortement significatif" if p_h3 < 0.001 else ("significatif" if p_h3 < 0.05 else "non significatif")
        conf_h3 = p_h3 < 0.05
        bc3 = "#D1FAE5" if conf_h3 else "#FEE2E2"
        brd3 = SUCCESS if conf_h3 else "#EF4444"
        st.markdown(f"""
        <div style="background:{bc3}; border-left:4px solid {brd3};
                    border-radius:8px; padding:14px 18px; margin:10px 0 16px; font-size:0.875rem;">
            <strong>{"✅" if conf_h3 else "❌"} H3 — Résultat</strong><br>
            Chi-deux : <strong>χ² = {chi2_h3:.1f}</strong> · ddl = {dof_h3} · p-value : <strong>{p_h3:.2e}</strong> ({sig_h3})<br>
            {"L'âge est significativement associé au mode de financement choisi."
             if conf_h3 else
             "Aucune association significative détectée entre âge et financement."}
        </div>""", unsafe_allow_html=True)

        insight(
            "Une association significative permet d'adapter le discours commercial dès la prise de contact. "
            "<br><br>"
            "👉 <strong>Usage marketing</strong> : si H3 est confirmée, mettre en avant la LOA pour les jeunes "
            "et les offres comptant/crédit pour les seniors dans les communications.",
            title="💡 Interprétation H3"
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ACM
        section("ACM — Biplot des associations entre modalités")

        insight(
            "L'<strong>Analyse des Correspondances Multiples (ACM)</strong> synthétise H2 et H3 en une seule visualisation. "
            "Les <strong>modalités proches</strong> sur le biplot ont tendance à co-apparaître dans les mêmes transactions. "
            "Les <strong>modalités opposées</strong> s'excluent mutuellement.",
            title="ℹ️ Comment lire le biplot ACM ?"
        )

        @st.cache_data
        def run_mca(data):
            d = data.copy()
            d["FI_TYPE"]     = d["FI_TYPE"].fillna("Comptant")
            d["TRANCHE_AGE"] = pd.cut(
                d["AGE"], bins=[17, 25, 35, 45, 55, 65, 100],
                labels=["18-25","26-35","36-45","46-55","56-65","65+"]
            )
            d["TRANCHE_PRIX"] = pd.cut(
                d["PRIX_VENTE_TTC_COM"],
                bins=[0, 15000, 20000, 25000, 100000],
                labels=["<15k","15-20k","20-25k",">25k"]
            )
            cols = ["TRANCHE_AGE","TRANCHE_PRIX","GENDER",
                    "VEHICULE_CATEGORIE","FI_TYPE","VEHICULE_ENERGIE","VEHICULE_TYPE"]
            mca_data = d[cols].dropna().astype(str)
            mca = prince.MCA(n_components=2, random_state=42)
            mca.fit(mca_data)
            col_coords = mca.column_coordinates(mca_data).reset_index()
            col_coords.columns = ["modality", "Dim1", "Dim2"]
            col_coords["variable"] = col_coords["modality"].str.split("__").str[0]
            col_coords["label"]    = col_coords["modality"].str.split("__").str[1]
            pct_var = mca.percentage_of_variance_
            return col_coords, pct_var

        col_coords, pct_var = run_mca(dff)

        var_labels = {
            "TRANCHE_AGE":        "Tranche d'âge",
            "TRANCHE_PRIX":       "Tranche de prix",
            "GENDER":             "Genre",
            "VEHICULE_CATEGORIE": "Catégorie",
            "FI_TYPE":            "Financement",
            "VEHICULE_ENERGIE":   "Énergie",
            "VEHICULE_TYPE":      "Type (VN/VO)",
        }
        col_coords["variable_label"] = col_coords["variable"].map(var_labels).fillna(col_coords["variable"])

        fig_mca = px.scatter(
            col_coords, x="Dim1", y="Dim2",
            color="variable_label",
            text="label",
            color_discrete_sequence=CHART_COLORS,
            labels={
                "Dim1": f"Dimension 1 ({pct_var[0]:.1f}% de variance)",
                "Dim2": f"Dimension 2 ({pct_var[1]:.1f}% de variance)",
                "variable_label": "Variable"
            }
        )
        fig_mca.update_traces(
            textposition="top center",
            marker=dict(size=10),
            textfont=dict(size=11)
        )
        fig_mca.add_hline(y=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig_mca.add_vline(x=0, line_dash="dot", line_color=BORDER, line_width=1)
        fig_mca.update_layout(legend_title="Variable")
        st.plotly_chart(chart_defaults(fig_mca, h=520), use_container_width=True)

        insight(
            "<strong>Comment exploiter ce biplot ?</strong><br>"
            "• Les modalités d'une même variable (même couleur) se répartissent selon leur profil d'achat.<br>"
            "• Les modalités de variables différentes proches = associations fortes (ex : Crédit + VO + 26-35 ans).<br>"
            "• Les modalités opposées au centre = profils distincts.<br><br>"
            "👉 Chaque cluster visuel de modalités = un profil client typique à cibler avec un message cohérent.",
            title="💡 Synthèse ACM"
        )


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
    AUCUNE_PREF = "Pas de préférence"
    with col2:
        r_gender  = st.selectbox("Genre", ["H","F"], format_func=lambda x: "Homme" if x=="H" else "Femme")
        r_energie = st.selectbox("Énergie préférée", [AUCUNE_PREF] + sorted(df["VEHICULE_ENERGIE"].dropna().unique().tolist()))
    with col3:
        r_categorie = st.selectbox("Catégorie souhaitée", [AUCUNE_PREF] + sorted(df["VEHICULE_CATEGORIE"].dropna().unique().tolist()))
        r_boite     = st.selectbox("Boîte de vitesse", [AUCUNE_PREF, "Boite manuelle", "Boite auto."])
    r_km_max = st.slider("Kilométrage maximum (km)", 0, 160000, 50000, step=5000)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Trouver les véhicules adaptés", type="primary", use_container_width=True):
        with st.spinner("Recherche en cours…"):
            def safe_enc(le, val):
                try: return le.transform([val])[0]
                except: return 0

            g_enc = safe_enc(le_g, r_gender)

            # Pour "Pas de préférence" : on passe la moyenne de la feature (→ 0 après scaling = neutre)
            e_enc = reco_scaler.mean_[4] if r_energie   == AUCUNE_PREF else safe_enc(le_e, r_energie)
            c_enc = reco_scaler.mean_[5] if r_categorie == AUCUNE_PREF else safe_enc(le_c, r_categorie)
            b_enc = reco_scaler.mean_[6] if r_boite     == AUCUNE_PREF else safe_enc(le_b, r_boite)

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

                pref_energie  = r_energie   if r_energie   != AUCUNE_PREF else "toutes énergies"
                pref_cat      = r_categorie if r_categorie != AUCUNE_PREF else "toutes catégories"
                insight(f"""Les véhicules listés correspondent à ceux achetés par des clients au profil similaire dans l'historique
                        (<strong>{r_age} ans, budget {r_budget:,} €, {pref_energie} · {pref_cat}</strong>).
                        Le modèle #1 est celui le plus fréquemment acheté parmi les 20 clients les plus proches.
                        <br><br>
                        👉 <strong>Pour l'équipe commerciale</strong> : utilisez ces résultats pour personnaliser
                        les emails de nurturing, les relances ou les landing pages produit.""",
                        title="💡 Comment utiliser ces résultats ?")

                # Graphique comparatif du prix des véhicules recommandés
                section("Comparatif prix des véhicules recommandés vs votre budget")
                fig_reco = px.bar(
                    reco, x="VEHICULE_MODELE", y="prix_moyen",
                    color="VEHICULE_MARQUE",
                    color_discrete_sequence=CHART_COLORS,
                    text=reco["prix_moyen"].apply(lambda x: f"{int(x):,} €"),
                    labels={"VEHICULE_MODELE": "Modèle", "prix_moyen": "Prix moyen (€)", "VEHICULE_MARQUE": "Marque"}
                )

                fig_reco.update_traces(
                    textposition="outside",
                    hovertemplate=
                        "<br>Marque : <b>%{fullData.name}</b>"
                        "<br>Modèle : <b>%{x}</b>"
                        "<br>Prix moyen : <b>%{y:,.0f} €</b>"
                        "<extra></extra>"
                )

                fig_reco.add_hline(
                    y=r_budget, line_dash="dot", line_color="#EF4444",
                    annotation_text=f"Budget max : {r_budget:,} €",
                    annotation_font=dict(color="#EF4444")
                )
                st.plotly_chart(chart_defaults(fig_reco, h=380), use_container_width=True)
