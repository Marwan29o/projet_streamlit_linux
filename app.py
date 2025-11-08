# app.py — MoSEF Auto Marketing (4 onglets, thème sombre + palette unifiée)
import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier

st.set_page_config(page_title="MoSEF Auto – Marketing", layout="wide")

# ===================== CSS THEME =====================
st.markdown("""
<style>
:root{
  --bg-app:#0B0C10;          /* fond global */
  --bg-panel:#1A1D23;        /* panneaux / zones graphiques */
  --bg-sidebar:#141922;      /* sidebar */
  --grid:#2D3142;            /* quadrillage */
  --text:#E8E9F3;            /* texte principal */
  --muted:#B0B6C3;           /* texte secondaire */

  /* KPI cards */
  --kpi-bg:#1E2738;          /* bleu nuit pour les cartes KPI */
  --kpi-border:#253146;      /* bordure KPI */
  --kpi-shadow:0 6px 16px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
}

/* Global */
html, body, [data-testid="stAppViewContainer"]{background:var(--bg-app); color:var(--text);}
[data-testid="stHeader"]{background:transparent;}
h1,h2,h3,h4,h5,h6{color:var(--text);}
p,span,li,small,label{color:var(--muted);}
strong,b{color:var(--text);}
[data-testid="stToolbar"]{display:none;} /* optionnel */

/* Sidebar */
[data-testid="stSidebar"]{background:var(--bg-sidebar); border-right:1px solid #202534;}
[data-testid="stSidebar"] *{color:var(--text);}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] small{color:var(--muted);}

/* Tabs */
.stTabs [role="tablist"]{border-bottom:1px solid #202534;}
.stTabs [role="tab"]{color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--text); background:#10151d; border-radius:8px 8px 0 0;}

/* KPI cards (st.metric) */
div[data-testid="stMetric"]{
  background:linear-gradient(180deg, var(--kpi-bg) 0%, var(--bg-panel) 100%);
  border:1px solid var(--kpi-border); border-radius:14px; padding:14px 16px;
  box-shadow:var(--kpi-shadow);
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"]{
  color:var(--muted); font-weight:600; letter-spacing:.2px;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{
  color:var(--text); font-weight:800; letter-spacing:.3px;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"]{
  font-weight:700; padding:2px 8px; border-radius:999px; background:rgba(255,255,255,0.06);
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"]:contains("+"){color:#19c37d;}
div[data-testid="stMetric"] [data-testid="stMetricDelta"]:not(:contains("+")){color:#ff5a66;}
div[data-testid="stMetric"] svg{display:none;} /* optionnel: masque la flèche */

/* Inputs & placeholders */
.stTextInput input, .stNumberInput input, .stTextArea textarea {color:var(--text);}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {color:#C8CED9; opacity:.75;}

/* ========== Select / MultiSelect (fond sombre partout) ========== */
.stSelectbox > div[data-baseweb="select"] > div,
.stMultiSelect > div[data-baseweb="select"] > div {
  background: var(--bg-panel) !important;
  border: 1px solid #202534 !important;
  border-radius: 10px !important;
}
.stSelectbox [data-baseweb="select"] *,
.stMultiSelect [data-baseweb="select"] * {
  color: var(--text) !important;
}
.stSelectbox [data-baseweb="select"] [class*="placeholder"],
.stMultiSelect [data-baseweb="select"] [class*="placeholder"] {
  color: #C8CED9 !important; opacity: .8 !important;
}
.stSelectbox [data-baseweb="popover"],
.stMultiSelect [data-baseweb="popover"],
.stSelectbox [role="listbox"],
.stMultiSelect [role="listbox"] {
  background: var(--bg-panel) !important;
  color: var(--text) !important;
  border: 1px solid #202534 !important;
  border-radius: 10px !important;
}
.stSelectbox [role="option"], .stMultiSelect [role="option"] { color: var(--text) !important; }
.stSelectbox > div[data-baseweb="select"] > div:hover,
.stMultiSelect > div[data-baseweb="select"] > div:hover { border-color: #41506a !important; }
.stSelectbox > div[data-baseweb="select"] > div:focus-within,
.stMultiSelect > div[data-baseweb="select"] > div:focus-within { outline: 0; border-color: #2B59C3 !important; }
.stSelectbox [aria-disabled="true"] > div,
.stMultiSelect [aria-disabled="true"] > div { background: #18202c !important; color: var(--muted) !important; }

/* Sliders Streamlit en bleu */
[data-baseweb="slider"] div[role="slider"]{
  background: #2B59C3 !important; border: 2px solid #1b2534 !important;
}
[data-baseweb="slider"] div[aria-hidden="true"]{
  background: linear-gradient(90deg, #2B59C3 0%, #2B59C3 100%) !important;
}

/* Expander / tables */
div[data-testid="stExpander"]{background:var(--bg-panel); border:1px solid #202534;}
.stDataFrame, .stTable {color:var(--text);}

/* Plotly toolbars/labels */
.js-plotly-plot, .modebar-group * { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ===================== PALETTE UNIFIÉE =====================
# Palette du donut (bleu foncé / bleu clair / rouge / saumon)
COLOR = {
    "BLUE_DARK":  "#2B59C3",  # Concession
    "BLUE_LIGHT": "#9FC2FF",  # Web
    "RED":        "#FF6B6B",  # Event
    "SALMON":     "#F7AFAF",  # Partenaire
    "GRID":       "#2D3142",
    "PANEL":      "#1A1D23",
    "APP":        "#0B0C10",
    "TEXT":       "#E8E9F3",
}
# Séquence par défaut (qualitative)
px.defaults.color_discrete_sequence = [
    COLOR["BLUE_DARK"], COLOR["BLUE_LIGHT"], COLOR["RED"], COLOR["SALMON"]
]
# Échelles continues bleues
BLUE_CONT = ["#0b111a", "#1c2c56", COLOR["BLUE_DARK"], "#6C93F2", COLOR["BLUE_LIGHT"]]
px.defaults.color_continuous_scale = BLUE_CONT

def apply_theme(fig):
    fig.update_layout(
        plot_bgcolor=COLOR["PANEL"],
        paper_bgcolor=COLOR["APP"],
        font=dict(color=COLOR["TEXT"]),
        xaxis=dict(gridcolor=COLOR["GRID"], showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=COLOR["GRID"], showgrid=True, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color=COLOR["TEXT"]))
    )
    return fig

# Couleurs fixes lorsqu'on affiche la dimension "channel"
CHANNEL_COLORS = {
    "Concession": COLOR["BLUE_DARK"],
    "Web":        COLOR["BLUE_LIGHT"],
    "Event":      COLOR["RED"],
    "Partenaire": COLOR["SALMON"],
}

# ===================== CONFIG =====================
# Priorité : DATA_URL > GDRIVE_FILE_ID > fallback local
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "").strip()
DATA_URL = os.getenv("DATA_URL", "").strip()

if DATA_URL:
    DATA_PATH = DATA_URL
elif GDRIVE_FILE_ID:
    DATA_PATH = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
else:
    # Fallback local (à adapter si besoin)
    st.error("Aucune source de données définie. Ajoute GDRIVE_FILE_ID ou DATA_URL dans .env.")

# ===================== DATA =====================
@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    # --- Cas 1 : URL (Google Drive / HTTP)
    if path.startswith(("http://", "https://")):
        st.info("Chargement des données depuis une source externe 🌐")
        r = requests.get(path, timeout=30)
        r.raise_for_status()
        # Détection simple du séparateur (',' vs ';')
        head = r.text[:2048]
        sep_guess = ";" if head.count(";") > head.count(",") else ","
        df = pd.read_csv(io.StringIO(r.text), sep=sep_guess)
    else:
        # --- Cas 2 : fichier local
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fichier introuvable: {path}. "
                "Définis DATA_URL ou GDRIVE_FILE_ID dans ton .env."
            )
        # Si ton CSV local est en FR, ajuste sep/decimal si nécessaire
        df = pd.read_csv(path)  # ex: pd.read_csv(path, sep=';', decimal=',')

    # --- Traitements communs
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df = df.sort_values("date")

    return df.reset_index(drop=True)

def format_int(x): 
    return f"{int(x):,}".replace(",", " ")

def pct(x): 
    return f"{100*x:.2f}%"

# Appel de la fonction
df = load_data(DATA_PATH)

# ===================== SIDEBAR =====================
st.sidebar.header("🎛️ Filtres")
months = sorted(df["month"].unique())
period = st.sidebar.select_slider("📅 Période", options=months, value=(months[0], months[-1]))
model_sel = st.sidebar.selectbox("🚗 Modèle", ["(Tous)"] + sorted(df["model"].unique()))
region_sel = st.sidebar.selectbox("📍 Région", ["(Toutes)"] + sorted(df["region"].unique()))
channel_sel = st.sidebar.selectbox("📢 Canal", ["(Tous)"] + sorted(df["channel"].unique()))
st.sidebar.markdown("---"); st.sidebar.caption("MoSEF Auto © 2025")

# ===================== TABS =====================
tab_dash, tab_comp, tab_behav, tab_ml = st.tabs(
    ["📊 Vue d'ensemble", "⚖️ Analyse Comparative", "🔍 Analyse Comportementale", "🤖 Modèle ML"]
)

# ===================== PAGE 1 — DASHBOARD =====================
with tab_dash:
    st.title("MoSEF Auto – Dashboard Marketing")
    mask = (df["month"] >= period[0]) & (df["month"] <= period[1])
    if model_sel != "(Tous)":   mask &= df["model"] == model_sel
    if region_sel != "(Toutes)": mask &= df["region"] == region_sel
    if channel_sel != "(Tous)":  mask &= df["channel"] == channel_sel
    dff = df.loc[mask].copy()

    # KPIs
    n_months = max(1, months.index(period[1]) - months.index(period[0]) + 1)
    prev_end_idx = months.index(period[0]) - 1
    if prev_end_idx >= n_months - 1:
        prev_start_idx = prev_end_idx - (n_months - 1)
        prev_range = (months[prev_start_idx], months[prev_end_idx])
        prev_mask = (df["month"] >= prev_range[0]) & (df["month"] <= prev_range[1])
        if model_sel != "(Tous)":   prev_mask &= df["model"] == model_sel
        if region_sel != "(Toutes)": prev_mask &= df["region"] == region_sel
        if channel_sel != "(Tous)":  prev_mask &= df["channel"] == channel_sel
        dff_prev = df.loc[prev_mask]
    else:
        dff_prev = pd.DataFrame(columns=dff.columns)

    leads = len(dff)
    sales = int(dff["converted"].sum())
    conv = sales / leads if leads else 0.0
    price_avg = dff["price"].mean() if leads else 0.0

    def delta_val(curr, prev):
        if prev is None or prev == 0: return None
        return (curr - prev) / prev

    if not dff_prev.empty:
        leads_prev = len(dff_prev)
        sales_prev = int(dff_prev["converted"].sum())
        conv_prev = sales_prev / leads_prev if leads_prev else None
        price_prev = dff_prev["price"].mean() if leads_prev else None
    else:
        leads_prev = sales_prev = None
        conv_prev = price_prev = None

    st.markdown("### 📊 Indicateurs clés")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Leads", format_int(leads),
                  f"{delta_val(leads, leads_prev)*100:+.1f}%" if delta_val(leads, leads_prev) else None)
    with kpi2:
        st.metric("Ventes", format_int(sales),
                  f"{delta_val(sales, sales_prev)*100:+.1f}%" if delta_val(sales, sales_prev) else None)
    with kpi3:
        st.metric("Taux conversion", pct(conv),
                  f"{delta_val(conv, conv_prev)*100:+.1f}%" if delta_val(conv, conv_prev) else None)
    with kpi4:
        st.metric("Prix moyen", f"{price_avg:,.0f} €".replace(",", " "),
                  f"{delta_val(price_avg, price_prev)*100:+.1f}%" if delta_val(price_avg, price_prev) else None)

    # Tendances (bleu foncé & rouge)
    st.markdown("### 📈 Évolution dans le temps")
    m_agg = dff.groupby("month", as_index=False).agg(
        leads=("converted","count"), sales=("converted","sum"), avg_price=("price","mean")
    ).sort_values("month")
    fig_trend = px.line(m_agg, x="month", y=["leads","sales"], markers=True,
                        color_discrete_sequence=[COLOR["BLUE_DARK"], COLOR["RED"]])
    apply_theme(fig_trend)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Segments
    st.markdown("### 🎯 Analyse par segment")
    col_left, col_right = st.columns(2)
    with col_left:
        bc = dff[dff["converted"]==1].groupby("channel", as_index=False)["converted"].count().rename(columns={"converted":"sales"})
        fig_donut = px.pie(
            bc, names="channel", values="sales", hole=0.5,
            color="channel", color_discrete_map=CHANNEL_COLORS
        )
        apply_theme(fig_donut)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label',
                                textfont=dict(color=COLOR["APP"], size=12, family='Arial Black'))
        st.plotly_chart(fig_donut, use_container_width=True)
    with col_right:
        bm = dff[dff["converted"]==1].groupby("model", as_index=False)["converted"].count() \
             .rename(columns={"converted":"sales"}).sort_values("sales", ascending=True)
        fig_bar = px.bar(bm, x="sales", y="model", orientation="h",
                         color_discrete_sequence=[COLOR["BLUE_DARK"]])
        apply_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

# ===================== PAGE 2 — ANALYSE COMPARATIVE =====================
with tab_comp:
    st.title("⚖️ Analyse Comparative")
    st.markdown("Comparez les performances entre segments pour identifier les combinaisons gagnantes")

    # Filtres locaux
    dimension1 = st.selectbox("Dimension à comparer",
                              ["Modèle", "Canal", "Région", "Type de carburant"], index=0, key="dim1")
    dim_map = {"Modèle": "model", "Canal": "channel", "Région": "region", "Type de carburant": "fuel_type"}
    col_name = dim_map[dimension1]
    available_values = sorted(dff[col_name].unique())
    col_sel1, col_sel2, _ = st.columns([1,1,2])
    with col_sel1: segment1 = st.selectbox("Segment 1", available_values, index=0, key="seg1")
    with col_sel2: segment2 = st.selectbox("Segment 2", available_values, index=min(1, len(available_values)-1), key="seg2")

    seg1_data = dff[dff[col_name] == segment1]
    seg2_data = dff[dff[col_name] == segment2]
    def calc_metrics(data):
        return {
            "leads": len(data),
            "ventes": int(data["converted"].sum()),
            "taux": (data["converted"].mean() * 100) if len(data) > 0 else 0,
            "prix_moyen": data["price"].mean() if len(data) > 0 else 0,
            "remise_moy": (data["discount_rate"].mean() * 100) if len(data) > 0 else 0,
            "age_moyen": data["customer_age"].mean() if len(data) > 0 else 0,
            "test_drive_rate": (data["test_drive"].mean() * 100) if len(data) > 0 else 0
        }
    m1, m2 = calc_metrics(seg1_data), calc_metrics(seg2_data)

    col_m1, col_vs, col_m2 = st.columns([2, 0.3, 2])
    with col_m1:
        st.markdown(f"#### 🔵 {segment1}")
        st.metric("Leads", f"{m1['leads']:,}".replace(",", " "))
        st.metric("Ventes", f"{m1['ventes']:,}".replace(",", " "))
        st.metric("Taux conversion", f"{m1['taux']:.2f}%")
        st.metric("Prix moyen", f"{m1['prix_moyen']:,.0f} €".replace(",", " "))
        st.metric("Remise moyenne", f"{m1['remise_moy']:.1f}%")
        st.metric("Taux essai", f"{m1['test_drive_rate']:.1f}%")
    with col_vs: st.markdown("#### VS")
    with col_m2:
        st.markdown(f"#### 🟢 {segment2}")
        delta_leads = ((m2['leads'] - m1['leads']) / m1['leads'] * 100) if m1['leads'] > 0 else 0
        delta_ventes = ((m2['ventes'] - m1['ventes']) / m1['ventes'] * 100) if m1['ventes'] > 0 else 0
        delta_taux = m2['taux'] - m1['taux']
        delta_prix = ((m2['prix_moyen'] - m1['prix_moyen']) / m1['prix_moyen'] * 100) if m1['prix_moyen'] > 0 else 0
        delta_remise = m2['remise_moy'] - m1['remise_moy']
        delta_test = m2['test_drive_rate'] - m1['test_drive_rate']
        st.metric("Leads", f"{m2['leads']:,}".replace(",", " "), f"{delta_leads:+.1f}%")
        st.metric("Ventes", f"{m2['ventes']:,}".replace(",", " "), f"{delta_ventes:+.1f}%")
        st.metric("Taux conversion", f"{m2['taux']:.2f}%", f"{delta_taux:+.2f}pp")
        st.metric("Prix moyen", f"{m2['prix_moyen']:,.0f} €".replace(",", " "), f"{delta_prix:+.1f}%")
        st.metric("Remise moyenne", f"{m2['remise_moy']:.1f}%", f"{delta_remise:+.1f}pp")
        st.metric("Taux essai", f"{m2['test_drive_rate']:.1f}%", f"{delta_test:+.1f}pp")

    st.divider()
    st.markdown("### 🔥 Matrice de performance : Canal × Modèle")
    heatmap_data = dff.groupby(["channel", "model"]).agg(
        leads=("converted", "count"), conversions=("converted", "sum")
    ).reset_index()
    heatmap_data["taux"] = (heatmap_data["conversions"] / heatmap_data["leads"] * 100).round(2)
    heatmap_pivot = heatmap_data.pivot(index="channel", columns="model", values="taux")
    fig_heatmap = px.imshow(
        heatmap_pivot, labels=dict(x="Modèle", y="Canal", color="Taux (%)"),
        x=heatmap_pivot.columns, y=heatmap_pivot.index,
        color_continuous_scale=BLUE_CONT, aspect="auto", text_auto=".1f"
    )
    apply_theme(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ===================== PAGE 3 — ANALYSE COMPORTEMENTALE =====================
with tab_behav:
    st.title("🔍 Analyse Comportementale")
    col_funnel, col_campaigns = st.columns(2)
    with col_funnel:
        st.markdown("### 🎯 Funnel de conversion")
        total_leads = len(dff)
        with_testdrive = len(dff[dff["test_drive"] == 1])
        with_competitor = len(dff[dff["competitor_quote"] == 0])
        converted = len(dff[dff["converted"] == 1])
        funnel_data = pd.DataFrame({
            "Étape": ["Leads totaux", "Essai réalisé", "Sans concurrent", "Convertis"],
            "Nombre": [total_leads, with_testdrive, with_competitor, converted],
            "Taux": [100, (with_testdrive/total_leads)*100, (with_competitor/total_leads)*100, (converted/total_leads)*100]
        })
        fig_funnel = px.funnel(funnel_data, x="Nombre", y="Étape", text="Taux",
                               color_discrete_sequence=[COLOR["BLUE_DARK"]])
        apply_theme(fig_funnel)
        fig_funnel.update_traces(texttemplate='%{text:.1f}%', textposition='inside',
                                 textfont=dict(size=14, color=COLOR["APP"], family='Arial Black'))
        st.plotly_chart(fig_funnel, use_container_width=True)
    with col_campaigns:
        st.markdown("### 📢 Impact des campagnes marketing")
        camp_analysis = dff.groupby("campaigns_seen").agg(
            leads=("converted","count"), conversions=("converted","sum")
        ).reset_index()
        camp_analysis["taux"] = (camp_analysis["conversions"]/camp_analysis["leads"]*100).round(2)
        fig_camp = px.line(camp_analysis, x="campaigns_seen", y="taux", markers=True,
                           color_discrete_sequence=[COLOR["RED"]])
        apply_theme(fig_camp)
        fig_camp.update_traces(marker=dict(size=10, line=dict(width=2, color=COLOR["APP"])),
                               line=dict(width=4))
        st.plotly_chart(fig_camp, use_container_width=True)

    col_age, col_discount = st.columns(2)
    with col_age:
        st.markdown("### 👥 Conversion par tranche d'âge")
        dff['age_group'] = pd.cut(dff['customer_age'], bins=[0,25,35,45,55,100],
                                  labels=['18-25','26-35','36-45','46-55','56+'])
        age_analysis = dff.groupby("age_group", observed=True).agg(
            leads=("converted","count"), conversions=("converted","sum")
        ).reset_index()
        age_analysis["taux"] = (age_analysis["conversions"]/age_analysis["leads"]*100).round(2)
        fig_age = px.bar(age_analysis, x="age_group", y="taux", text="taux",
                         color_discrete_sequence=[COLOR["BLUE_DARK"]])
        apply_theme(fig_age)
        fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                              textfont=dict(size=13, color=COLOR["TEXT"], family='Arial Black'))
        st.plotly_chart(fig_age, use_container_width=True)
    with col_discount:
        st.markdown("### 💰 Sweet spot de remise")
        dff['discount_bucket'] = pd.cut(dff['discount_rate'], bins=[0,0.05,0.10,0.15,0.20,1],
                                        labels=['0-5%','5-10%','10-15%','15-20%','20%+'])
        discount_analysis = dff.groupby("discount_bucket", observed=True).agg(
            leads=("converted","count"), conversions=("converted","sum")
        ).reset_index()
        discount_analysis["taux"] = (discount_analysis["conversions"]/discount_analysis["leads"]*100).round(2)
        fig_discount = px.scatter(discount_analysis, x="discount_bucket", y="taux",
                                  size="leads", color="taux",
                                  color_continuous_scale=BLUE_CONT, size_max=60)
        apply_theme(fig_discount)
        fig_discount.update_traces(marker=dict(line=dict(width=2, color=COLOR["TEXT"])))
        fig_discount.update_layout(showlegend=False)
        st.plotly_chart(fig_discount, use_container_width=True)

    st.caption("💡 Ces insights permettent d'optimiser la stratégie marketing et commerciale")

# ===================== PAGE 4 — MODÈLE ML =====================
with tab_ml:
    st.title("Modèle de propension de conversion (CatBoost)")
    st.write("Modèle : **CatBoostClassifier** - Gère nativement les variables catégorielles et fournit une importance des variables précise sans besoin de one-hot encoding.")
    mask_ml = (df["month"] >= period[0]) & (df["month"] <= period[1])
    if model_sel != "(Tous)":   mask_ml &= df["model"] == model_sel
    if region_sel != "(Toutes)": mask_ml &= df["region"] == region_sel
    if channel_sel != "(Tous)":  mask_ml &= df["channel"] == channel_sel
    dml = df.loc[mask_ml].copy()

    target = "converted"
    feat_num = ["price","discount_rate","campaigns_seen","customer_age"]
    feat_cat = ["channel","model","fuel_type","income_band","test_drive","competitor_quote"]
    all_features = feat_num + feat_cat

    st.write(f"Échantillon : {len(dml):,} lignes".replace(",", " "))
    if len(dml) < 1500 or dml[target].nunique() < 2:
        st.warning("Élargis la période / enlève des filtres : il n'y a pas assez de diversité pour entraîner un modèle.")
        st.stop()

    X = dml[all_features].copy()
    y = dml[target].astype(int).copy()
    for col in feat_cat: X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    cat_features_idx = [all_features.index(col) for col in feat_cat]

    with st.spinner("Entraînement du modèle CatBoost..."):
        model = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='AUC',
            random_seed=42, verbose=False, cat_features=cat_features_idx
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.2).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.divider()
    st.subheader("📊 Importance des variables (CatBoost native)")

    # Importance des variables — barres bleu foncé + fond sombre
    feature_importance = model.get_feature_importance()
    imp_df = pd.DataFrame({"feature": all_features, "importance": feature_importance}).sort_values("importance", ascending=False)
    fig_imp = px.bar(
        imp_df, x="importance", y="feature", orientation="h",
        title="Importance des variables (toutes)",
        labels={"importance":"Score d'importance", "feature":"Variable"},
        color_discrete_sequence=[COLOR["BLUE_DARK"]]
    )
    fig_imp.update_traces(marker=dict(line=dict(width=0)))
    fig_imp.update_layout(
        plot_bgcolor=COLOR["PANEL"], paper_bgcolor=COLOR["APP"],
        font=dict(color=COLOR["TEXT"]),
        xaxis=dict(gridcolor=COLOR["GRID"], showgrid=True, zeroline=False),
        yaxis=dict(showgrid=False), yaxis_title=""
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("📋 Voir le tableau d'importance détaillé"):
        imp_df['importance_pct'] = (imp_df['importance'] / imp_df['importance'].sum() * 100).round(2)
        st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

    st.divider()
    st.subheader("🎯 Performance du modèle")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AUC", f"{auc:.3f}", help="Area Under Curve - capacité à distinguer les classes")
    k2.metric("Accuracy", f"{acc:.3f}", help="Taux de prédictions correctes")
    k3.metric("Échantillon test", format_int(len(X_test)), help="Nombre d'observations pour l'évaluation")
    k4.metric("Itérations", model.get_best_iteration(), help="Nombre d'itérations du modèle")

    with st.expander("📉 Voir la matrice de confusion"):
        cm_df = pd.DataFrame(cm, index=["Vrai 0","Vrai 1"], columns=["Prédit 0","Prédit 1"])
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Seuil à 0.2 pour capturer plus de leads potentiels (favorise le rappel)")

    st.divider()
    st.subheader("🧪 Scorer un lead")
    with st.form("score_form"):
        colA, colB, colC = st.columns(3)
        with colA:
            f_channel = st.selectbox("Canal", sorted(df["channel"].unique()))
            f_model = st.selectbox("Modèle", sorted(df["model"].unique()))
            f_fuel = st.selectbox("Énergie", sorted(df["fuel_type"].unique()))
        with colB:
            f_income = st.selectbox("Tranche de revenu", ["Low","Medium","High"])
            f_age = st.slider("Âge client", 18, 80, 42, 1)
            f_testdrive = st.selectbox("Essai réalisé ?", ["Non","Oui"])
        with colC:
            f_price = st.number_input("Prix proposé (€)", 8000, 120000, 30000, 500)
            f_discount = st.slider("Remise (%)", 0.0, 25.0, 5.0, 0.5) / 100.0
            f_campaigns = st.slider("Campagnes vues", 0, 10, 2, 1)
            f_competitor = st.selectbox("Devis concurrent ?", ["Non","Oui"])

        submitted = st.form_submit_button("Estimer la probabilité")
        if submitted:
            new = pd.DataFrame([{
                "price": f_price,
                "discount_rate": f_discount,
                "campaigns_seen": f_campaigns,
                "customer_age": f_age,
                "channel": str(f_channel),
                "model": str(f_model),
                "fuel_type": str(f_fuel),
                "income_band": str(f_income),
                "test_drive": str(1 if f_testdrive=="Oui" else 0),
                "competitor_quote": str(1 if f_competitor=="Oui" else 0),
            }])
            prob = float(model.predict_proba(new)[0,1])
            st.success(f"Probabilité de conversion estimée : **{prob*100:.1f}%**")
            if prob >= 0.7:
                st.info("🎯 Lead chaud ! Priorise ce contact.")
            elif prob >= 0.4:
                st.info("🔥 Lead tiède. Relance avec une offre ciblée.")
            else:
                st.info("❄️ Lead froid. Nurturing à long terme recommandé.")

    st.caption("CatBoost : modèle puissant qui gère nativement les variables catégorielles sans preprocessing complexe.")
