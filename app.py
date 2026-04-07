"""
Streamlit Web Application — Waiter's Tips Prediction System
============================================================
Premium dark-gold theme: Matte Black · Charcoal · Gold · Amber · Ivory
Icons: Lucide CDN (lucide-static via jsDelivr)

Run with:
    streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(BASE_DIR, 'src')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'tips.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
XAI_DIR     = os.path.join(RESULTS_DIR, 'explainability')

sys.path.insert(0, SRC_DIR)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tips Predictor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# ── Lucide CDN helper ─────────────────────────────────────────────────────────
_LUCIDE_CDN = "https://cdn.jsdelivr.net/npm/lucide-static@latest/icons"


def lucide(name: str, size: int = 16, css_class: str = "li") -> str:
    """
    Return an <img> tag loading the icon from the official Lucide CDN
    (lucide-static via jsDelivr). Color is applied via CSS class filters.
    """
    return (
        f'<img src="{_LUCIDE_CDN}/{name}.svg" '
        f'width="{size}" height="{size}" '
        f'class="{css_class}" alt="{name}" />'
    )


def pg_title(icon: str, text: str) -> str:
    """Page heading with a Lucide CDN icon."""
    return (
        f'<div class="pg-title">'
        f'{lucide(icon, size=34, css_class="li-page")}'
        f'{text}</div>'
    )


def sec_label(icon: str, text: str) -> str:
    """Section label with a Lucide CDN icon."""
    return (
        f'<span class="sec-label">'
        f'{lucide(icon, size=12, css_class="li-sec")}'
        f'{text}</span>'
    )


# ── Navigation config ─────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("predict",  "target",      "Predict Tip"),
    ("explorer", "bar-chart-2", "Data Explorer"),
    ("compare",  "layers",      "Model Comparison"),
    ("about",    "info",        "About"),
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Space+Mono:wght@400;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ═══════════════════════════════════════════════
   PALETTE
   Primary   Matte Black  #0B0B0B
   Secondary Charcoal     #1F2937
   Accent    Gold         #D4AF37
   Highlight Amber        #F59E0B
   Text      Ivory        #F9FAFB
═══════════════════════════════════════════════ */
:root {
    --gold:    #D4AF37;
    --amber:   #F59E0B;
    --gold-d:  #a8872a;
    --ivory:   #F9FAFB;
    --iv2:     #E5E7EB;
    --iv3:     #9CA3AF;
    --iv4:     #6B7280;
    --iv5:     #4B5563;
    --black:   #0B0B0B;
    --ch:      #1F2937;
    --ch2:     #263344;
    --ch3:     #374151;
    --rule:    1px solid #374151;
    --font-d:  'DM Serif Display', Georgia, serif;
    --font-m:  'Space Mono', 'Courier New', monospace;
    --font-b:  'DM Sans', system-ui, sans-serif;
}
* { box-sizing: border-box; }

/* ── App shell ── */
.stApp {
    background-color: var(--black) !important;
    color: var(--iv2) !important;
    font-family: var(--font-b) !important;
}

/* Grain overlay */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 99999;
}

[data-testid="stHeader"] { background: transparent !important; border-bottom: var(--rule) !important; }

/* Main content never overlaps sidebar — Streamlit handles the shift via its own
   sidebar width var; we just ensure smooth transition */
.main { transition: margin-left 0.3s ease !important; }
.main .block-container { padding: 2.5rem 3rem 4rem !important; max-width: 1360px !important; }

/* ══════════════════════════════════════════
   SIDEBAR  — no scroll, collapses cleanly
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: var(--ch) !important;
    border-right: 1px solid var(--ch3) !important;
}
/* Lock the inner scroll container so sidebar never scrolls */
[data-testid="stSidebar"] > div:first-child {
    overflow-y: hidden !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    padding-bottom: 1rem !important;
}
[data-testid="stSidebarContent"] {
    overflow-y: hidden !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span { font-family: var(--font-b) !important; color: var(--iv4) !important; font-size: 0.72rem !important; }
[data-testid="stSidebar"] .stSelectbox label { font-family: var(--font-b) !important; font-size: 0.6rem !important; font-weight: 600 !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; color: var(--iv5) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: var(--ch2) !important; border-color: var(--ch3) !important; border-radius: 0 !important; color: var(--ivory) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: var(--iv2) !important; font-family: var(--font-b) !important; font-size: 0.88rem !important; }
[data-testid="stSidebar"] hr { border-top: 1px solid var(--ch3) !important; }

/* ── Sidebar model status card ── */
.model-card { display: flex; align-items: center; gap: 0.65rem; padding: 0.75rem 1rem; background: var(--ch2); border: 1px solid var(--ch3); border-left: 2px solid var(--gold); margin-top: 0.5rem; }
.model-card.warn { border-left-color: var(--iv5); }
.model-card img { flex-shrink: 0; }
.model-card-name { font-family: var(--font-b); font-size: 0.8rem; font-weight: 600; color: var(--iv2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.model-card-status { font-family: var(--font-b); font-size: 0.6rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--gold); margin-top: 0.1rem; }
.model-card.warn .model-card-status { color: var(--iv5); }
.sb-footer { font-family: var(--font-b); font-size: 0.6rem; color: var(--iv5); letter-spacing: 0.1em; text-transform: uppercase; padding: 0.75rem 0 0; margin-top: auto; border-top: 1px solid var(--ch3); display: flex; align-items: center; gap: 0.4rem; }

/* ══════════════════════════════════════════
   LUCIDE CDN ICON FILTERS
   Icons are black SVG from CDN — CSS filter recolors them.
══════════════════════════════════════════ */

/* Nav icons — inactive: invert to white, dim to gray */
.nav-link .li {
    filter: invert(1);
    opacity: 0.35;
    flex-shrink: 0;
    transition: filter 0.15s, opacity 0.15s;
    display: inline-block;
    vertical-align: middle;
}
/* Nav icons — active/hover: gold */
.nav-link:hover .li,
.nav-link.active .li {
    filter: invert(75%) sepia(43%) saturate(612%) hue-rotate(5deg) brightness(95%) contrast(89%);
    opacity: 1;
}

/* Page title icons — ivory/white */
.li-page {
    filter: invert(1);
    opacity: 0.9;
    vertical-align: middle;
    margin-right: 0.55rem;
    margin-bottom: 0.4rem;
    display: inline-block;
}

/* Section label icons — gold */
.li-sec {
    filter: invert(75%) sepia(43%) saturate(612%) hue-rotate(5deg) brightness(95%) contrast(89%);
    opacity: 0.9;
    vertical-align: middle;
    margin-right: 0.35rem;
    display: inline-block;
}

/* Inline body icons — muted */
.li-muted {
    filter: invert(1);
    opacity: 0.4;
    vertical-align: middle;
    margin-right: 0.25rem;
    display: inline-block;
}

/* ══════════════════════════════════════════
   LUCIDE NAVIGATION
══════════════════════════════════════════ */
.lucide-nav { display: flex; flex-direction: column; margin: 1rem -1rem 0; border-top: 1px solid #374151; }

.nav-link {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.25rem;
    text-decoration: none !important;
    color: #6B7280 !important;
    border-bottom: 1px solid #374151;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    transition: color 0.15s, background 0.15s;
    cursor: pointer;
}
.nav-link:hover { color: #D4AF37 !important; background: rgba(212,175,55,0.05); text-decoration: none !important; }
.nav-link.active { color: #D4AF37 !important; font-weight: 600; border-left: 2px solid #D4AF37; padding-left: calc(1.25rem - 2px); background: rgba(212,175,55,0.07); }

/* ══════════════════════════════════════════
   BUTTONS — Gold gradient
══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #D4AF37, #F59E0B) !important;
    color: var(--black) !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--font-b) !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.1s, box-shadow 0.15s !important;
    box-shadow: 0 4px 14px rgba(212,175,55,0.25) !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(212,175,55,0.35) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Selectboxes ── */
[data-baseweb="select"] > div { background-color: var(--ch) !important; border: 1px solid var(--ch3) !important; border-radius: 0 !important; color: var(--ivory) !important; }
[data-baseweb="select"] span { color: var(--iv2) !important; font-family: var(--font-b) !important; font-size: 0.9rem !important; }
[data-baseweb="popover"] { background-color: var(--ch) !important; border: 1px solid var(--ch3) !important; border-radius: 0 !important; }
[data-baseweb="menu"] li { background-color: var(--ch) !important; color: var(--iv2) !important; font-family: var(--font-b) !important; font-size: 0.9rem !important; }
[data-baseweb="menu"] li:hover { background-color: var(--ch2) !important; color: var(--gold) !important; }

/* ── Sliders — gold thumb ── */
[data-testid="stSlider"] > div > div > div > div { background: var(--gold) !important; }

/* ── Number inputs ── */
[data-testid="stNumberInput"] input { background: var(--ch) !important; border: 1px solid var(--ch3) !important; border-radius: 0 !important; color: var(--ivory) !important; font-family: var(--font-m) !important; font-size: 0.9rem !important; cursor: text !important; }
[data-testid="stNumberInput"] button { cursor: pointer !important; }

/* ── Cursor rules ── */
/* Clickable: pointer */
[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] input,
[data-baseweb="menu"] li,
[data-testid="stSlider"] input,
[data-testid="stSlider"] > div,
[data-testid="stSlider"] > div > div,
[data-testid="stSlider"] > div > div > div,
[data-testid="stSlider"] > div > div > div > div,
[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stPills"] button,
[data-testid="stToggle"] [role="switch"],
.nav-link,
.stButton > button { cursor: pointer !important; }
/* Typing fields: text cursor */
input[type="text"],
input[type="number"],
textarea { cursor: text !important; }
/* Everything else: default arrow */
label,
p,
[data-testid="stMetric"],
[data-testid="stMarkdownContainer"] { cursor: default !important; }

/* ── Labels ── */
label, .stSlider label, .stSelectbox label, .stNumberInput label { font-family: var(--font-b) !important; font-size: 0.68rem !important; font-weight: 500 !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; color: var(--iv4) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] { background: var(--ch) !important; border: 1px solid var(--ch3) !important; border-top: 2px solid var(--gold) !important; padding: 1.25rem 1.5rem !important; border-radius: 0 !important; }
[data-testid="stMetricLabel"] p { font-family: var(--font-b) !important; font-size: 0.6rem !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; color: var(--iv4) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-m) !important; font-size: 1.6rem !important; font-weight: 700 !important; color: var(--ivory) !important; }

/* ── Tabs — gold indicator ── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--ch3) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--iv5) !important; font-family: var(--font-b) !important; font-size: 0.68rem !important; font-weight: 500 !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; border-bottom: 2px solid transparent !important; padding: 0.8rem 1.5rem !important; transition: color 0.12s !important; }
.stTabs [data-baseweb="tab"]:hover { color: var(--gold) !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--gold) !important; border-bottom-color: var(--gold) !important; font-weight: 700 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--ch3) !important; border-radius: 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] { border: 1px solid var(--ch3) !important; border-radius: 0 !important; background: var(--ch) !important; }
[data-testid="stExpander"] summary { font-family: var(--font-b) !important; font-size: 0.7rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--iv4) !important; padding: 0.85rem 1rem !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 0 !important; background: var(--ch) !important; border: 1px solid var(--ch3) !important; border-left: 2px solid var(--iv5) !important; font-family: var(--font-b) !important; font-size: 0.85rem !important; }
.stSuccess { border-left-color: var(--gold) !important; color: var(--gold) !important; }
.stWarning { border-left-color: var(--amber) !important; color: var(--amber) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── Divider ── */
hr { border: none !important; border-top: var(--rule) !important; margin: 2rem 0 !important; }

/* ── Images ── */
.stImage img { border: 1px solid var(--ch3) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--black); }
::-webkit-scrollbar-thumb { background: var(--ch3); }
::-webkit-scrollbar-thumb:hover { background: var(--gold-d); }

/* ── Markdown table ── */
.stMarkdown table { border-collapse: collapse; width: 100%; font-family: var(--font-b); font-size: 0.84rem; }
.stMarkdown th { font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--iv4); border-bottom: 1px solid var(--ch3); padding: 0.6rem 1rem; text-align: left; font-weight: 600; }
.stMarkdown td { color: var(--iv2); border-bottom: 1px solid var(--ch); padding: 0.65rem 1rem; }

/* ── Code ── */
.stCode, code { background: var(--ch) !important; border: 1px solid var(--ch3) !important; border-radius: 0 !important; color: var(--iv2) !important; font-family: var(--font-m) !important; }

/* ── Animations ── */
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* ════════════════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════════════════ */
.pg-title { font-family: 'DM Serif Display', Georgia, serif; font-size: 2.9rem; font-weight: 400; color: #F9FAFB; letter-spacing: -0.025em; line-height: 1.05; margin: 0 0 0.5rem 0; animation: fadeUp 0.35s ease both; display: flex; align-items: center; }
.pg-sub { font-family: 'DM Sans', sans-serif; font-size: 0.82rem; color: #6B7280; letter-spacing: 0.03em; margin: 0 0 2.5rem 0; animation: fadeUp 0.35s 0.06s ease both; }

.sec-label { font-family: 'DM Sans', sans-serif; font-size: 0.6rem; font-weight: 600; letter-spacing: 0.22em; text-transform: uppercase; color: #D4AF37; padding-bottom: 0.65rem; border-bottom: 1px solid #374151; margin: 2rem 0 1.25rem 0; display: flex; align-items: center; gap: 0.35rem; }

.pred-hero { background: linear-gradient(160deg, #1F2937 0%, #263344 100%); border: 1px solid #374151; padding: 3.5rem 2.5rem 3rem; text-align: center; position: relative; animation: fadeUp 0.4s ease both; backdrop-filter: blur(6px); box-shadow: 0 0 40px rgba(212,175,55,0.12), 0 8px 32px rgba(0,0,0,0.45); }
.pred-hero::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #D4AF37, #F59E0B, #D4AF37); }
.pred-eyebrow { font-family: 'DM Sans', sans-serif; font-size: 0.6rem; letter-spacing: 0.28em; text-transform: uppercase; color: #6B7280; margin-bottom: 1rem; }
.pred-num { font-family: 'Space Mono', monospace; font-size: 6rem; font-weight: 700; color: #D4AF37; line-height: 1; letter-spacing: -0.04em; margin: 0; text-shadow: 0 0 40px rgba(212,175,55,0.25); }
.pred-pct { font-family: 'DM Sans', sans-serif; font-size: 0.88rem; color: #6B7280; letter-spacing: 0.06em; margin-top: 0.75rem; }
.pred-rule { width: 1.5rem; height: 1px; background: #374151; margin: 1.75rem auto; }

.await-box { background: #111827; border: 1px dashed #374151; padding: 3.5rem 2rem; text-align: center; }
.await-box p { font-family: 'DM Sans', sans-serif; font-size: 0.7rem; color: #4B5563; letter-spacing: 0.18em; text-transform: uppercase; margin: 0; }

.bench { border: 1px solid #374151; border-left: 3px solid #D4AF37; padding: 0.9rem 1.25rem; font-family: 'DM Sans', sans-serif; font-size: 0.84rem; color: #E5E7EB; margin-top: 1rem; background: #1F2937; }
.bench.below { border-left-color: #4B5563; color: #6B7280; }

.best-strip { background: linear-gradient(135deg, #D4AF37, #F59E0B); color: #0B0B0B; font-family: 'DM Sans', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.9rem 1.5rem; margin-top: 1.5rem; display: flex; gap: 2.5rem; align-items: center; flex-wrap: wrap; }
.best-strip .dim { font-weight: 500; opacity: 0.65; font-size: 0.65rem; }

.sb-brand { font-family: 'DM Serif Display', Georgia, serif; font-size: 1.2rem; color: #D4AF37; line-height: 1.2; margin: 0.5rem 0 0.25rem; text-shadow: 0 0 20px rgba(212,175,55,0.3); }
.sb-tag { font-family: 'DM Sans', sans-serif; font-size: 0.6rem; color: #4B5563; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0; display: block; }

.ibox { background: #1F2937; border: 1px solid #374151; border-left: 2px solid #4B5563; padding: 0.9rem 1.2rem; font-family: 'DM Sans', sans-serif; font-size: 0.84rem; color: #6B7280; margin: 0.75rem 0; }
.ibox strong, .ibox code { color: #D4AF37; }

.pill { display: inline-block; font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #9CA3AF; border: 1px solid #374151; padding: 0.22rem 0.65rem; margin: 0.2rem 0.15rem; letter-spacing: 0.04em; }

.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #374151; border: 1px solid #374151; margin: 1.5rem 0; }
.stat-cell { background: #111827; padding: 1.75rem 1.5rem; border-top: 2px solid #D4AF37; }
.stat-lbl { font-family: 'DM Sans', sans-serif; font-size: 0.58rem; letter-spacing: 0.22em; text-transform: uppercase; color: #4B5563; margin-bottom: 0.5rem; }
.stat-val { font-family: 'DM Serif Display', Georgia, serif; font-size: 2rem; color: #D4AF37; line-height: 1.1; }
@media (max-width: 800px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 400px) { .stat-grid { grid-template-columns: 1fr; } }

/* ══════════════════════════════════════════
   st.pills  — Gold chip buttons
══════════════════════════════════════════ */
[data-testid="stPills"] [data-testid="stWidgetLabel"] p {
    font-family: var(--font-b) !important;
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--iv5) !important;
    margin-bottom: 0.45rem !important;
}
/* Pill chips wrapper */
[data-testid="stPills"] > div {
    gap: 0.4rem !important;
    flex-wrap: wrap !important;
}
/* Individual chip */
[data-testid="stPills"] button {
    background: #111827 !important;
    border: 1px solid var(--ch3) !important;
    border-radius: 2px !important;
    color: var(--iv4) !important;
    font-family: var(--font-b) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.9rem !important;
    transition: all 0.14s ease !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
}
[data-testid="stPills"] button:hover {
    border-color: var(--gold-d) !important;
    color: var(--iv2) !important;
    background: rgba(212,175,55,0.06) !important;
}
/* Selected chip — gold gradient */
[data-testid="stPills"] button[aria-pressed="true"],
[data-testid="stPills"] button[data-selected="true"],
[data-testid="stPills"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #D4AF37, #F59E0B) !important;
    border-color: transparent !important;
    color: #0B0B0B !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(212,175,55,0.35) !important;
}

/* ══════════════════════════════════════════
   st.toggle  — Gold track when ON
══════════════════════════════════════════ */
[data-testid="stToggle"] [data-testid="stWidgetLabel"] p,
[data-testid="stToggle"] label p {
    font-family: var(--font-b) !important;
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--iv5) !important;
}
/* OFF track */
[data-testid="stToggle"] [role="switch"] {
    background-color: var(--ch3) !important;
    transition: background 0.2s ease !important;
    border: none !important;
}
/* ON track — gold */
[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background: linear-gradient(90deg, #D4AF37, #F59E0B) !important;
}
/* Thumb (white dot) */
[data-testid="stToggle"] [role="switch"] > div {
    background: #F9FAFB !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
}
/* Value label beside toggle */
[data-testid="stToggle"] .stMarkdown p {
    font-family: var(--font-b) !important;
    font-size: 0.82rem !important;
    color: var(--iv3) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Encoding maps ──────────────────────────────────────────────────────────────
ENCODE = {
    'sex':    {'Female': 0, 'Male': 1},
    'smoker': {'No': 0, 'Yes': 1},
    'day':    {'Fri': 0, 'Sat': 1, 'Sun': 2, 'Thur': 3},
    'time':   {'Dinner': 0, 'Lunch': 1},
}

MODEL_FILES = {
    'Random Forest': 'random_forest.pkl',
    'ID3 Tree':      'id3_tree.pkl',
    'CART Tree':     'cart_tree.pkl',
}

SAMPLE_CASES = [
    {'total_bill': 25.50, 'sex': 'Male',   'smoker': 'No',  'day': 'Sat',  'time': 'Dinner', 'size': 2},
    {'total_bill': 48.27, 'sex': 'Female', 'smoker': 'Yes', 'day': 'Fri',  'time': 'Dinner', 'size': 4},
    {'total_bill': 15.04, 'sex': 'Male',   'smoker': 'No',  'day': 'Sun',  'time': 'Lunch',  'size': 3},
    {'total_bill': 35.83, 'sex': 'Female', 'smoker': 'No',  'day': 'Sat',  'time': 'Dinner', 'size': 2},
    {'total_bill': 10.34, 'sex': 'Male',   'smoker': 'Yes', 'day': 'Thur', 'time': 'Lunch',  'size': 1},
]
FEATURE_NAMES = ['total_bill', 'sex', 'smoker', 'day', 'time', 'size']

# ── Chart helpers ──────────────────────────────────────────────────────────────
_BG    = '#0B0B0B'
_CARD  = '#1F2937'
_BORD  = '#374151'
_GOLD  = '#D4AF37'
_AMBER = '#F59E0B'
_IVORY = '#F9FAFB'
_MUTED = '#6B7280'
_DIM   = '#4B5563'


def _style_ax(ax):
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_MUTED, labelsize=8.5, length=0)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color(_IVORY)
    ax.title.set_fontsize(11)
    for spine in ax.spines.values():
        spine.set_color(_BORD)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(color='#263344', linewidth=0.6, linestyle='--')
    ax.set_axisbelow(True)


def dark_fig(w=11, h=4, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    if nrows == 1 and ncols == 1:
        _style_ax(axes)
    elif nrows == 1 or ncols == 1:
        for ax in axes:
            _style_ax(ax)
    else:
        for row in axes:
            for ax in row:
                _style_ax(ax)
    return fig, axes


def rank_golds(n):
    if n == 1:
        return [_GOLD]
    fr, fg, fb = 0xD4, 0xAF, 0x37
    tr, tg, tb = 0x37, 0x41, 0x51
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(f'#{int(fr+(tr-fr)*t):02x}{int(fg+(tg-fg)*t):02x}{int(fb+(tb-fb)*t):02x}')
    return out


_GOLD_DIV = mcolors.LinearSegmentedColormap.from_list(
    'gold_div', [_CARD, _BORD, _MUTED, _GOLD, _AMBER]
)


# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODELS_DIR, MODEL_FILES.get(model_name, ''))
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def get_model_metrics():
    from data_preprocessing import DataPreprocessor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    preprocessor = DataPreprocessor()
    _, X_test, _, y_test, _, _ = preprocessor.preprocess_pipeline()
    metrics = {}
    for name in MODEL_FILES:
        m = load_model(name)
        if m is None:
            continue
        y_pred = m.predict(X_test)
        metrics[name] = {
            'R² Score': round(r2_score(y_test, y_pred), 4),
            'RMSE ($)': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'MAE ($)':  round(mean_absolute_error(y_test, y_pred), 4),
        }
    return metrics


@st.cache_resource
def load_scaler():
    path = os.path.join(MODELS_DIR, 'scaler.pkl')
    return joblib.load(path) if os.path.exists(path) else None


def encode_features(total_bill, sex, smoker, day, time, size):
    raw = np.array([[
        total_bill,
        ENCODE['sex'][sex],
        ENCODE['smoker'][smoker],
        ENCODE['day'][day],
        ENCODE['time'][time],
        size,
    ]])
    scaler = load_scaler()
    return scaler.transform(raw) if scaler is not None else raw


# ── Auto-retrain when data/tips.csv changes or models are missing ─────────────
def _data_hash():
    import hashlib
    if not os.path.exists(DATA_PATH):
        return None
    with open(DATA_PATH, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _needs_retrain():
    hash_file = os.path.join(MODELS_DIR, 'data_hash.txt')
    models_ready = all(
        os.path.exists(os.path.join(MODELS_DIR, fn))
        for fn in MODEL_FILES.values()
    )
    if not models_ready:
        return True
    if not os.path.exists(hash_file):
        return True
    with open(hash_file) as f:
        return f.read().strip() != _data_hash()


def _run_retrain():
    from data_preprocessing import DataPreprocessor
    from model_training import TipPredictor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, _, _ = preprocessor.preprocess_pipeline()
    predictor = TipPredictor()
    predictor.train_all_models(X_train, X_test, y_train, y_test, tune=False)
    predictor.save_all_models()
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'data_hash.txt'), 'w') as f:
        f.write(_data_hash())
    st.cache_data.clear()
    st.cache_resource.clear()


if 'models_checked' not in st.session_state:
    if _needs_retrain():
        with st.spinner('Data changed — retraining models on new dataset...'):
            _run_retrain()
    st.session_state['models_checked'] = True


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Lucide CDN nav via query-param anchor links
# ══════════════════════════════════════════════════════════════════════════════
page_id = st.query_params.get("page", "predict")

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">Waiter's<br>Tips Predictor</div>
    <span class="sb-tag">ML Prediction System</span>
    """, unsafe_allow_html=True)

    # Lucide CDN nav
    nav_html = '<nav class="lucide-nav">'
    for pid, icon_name, label in NAV_ITEMS:
        active_cls = "active" if page_id == pid else ""
        nav_html += (
            f'<a href="?page={pid}" target="_self" class="nav-link {active_cls}">'
            f'{lucide(icon_name, size=16)}'           # CDN <img>
            f'<span>{label}</span>'
            f'</a>'
        )
    nav_html += '</nav>'
    st.markdown(nav_html, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#374151;margin:0.75rem -1rem;"></div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Active Model", list(MODEL_FILES.keys()), index=0, label_visibility="visible")
    model = load_model(selected_model_name)
    if model:
        st.markdown(
            f'<div class="model-card">'
            f'<img src="{_LUCIDE_CDN}/check-circle.svg" width="14" height="14" '
            f'style="filter:invert(75%) sepia(43%) saturate(612%) hue-rotate(5deg) brightness(95%) contrast(89%);flex-shrink:0;">'
            f'<div><div class="model-card-name">{selected_model_name}</div>'
            f'<div class="model-card-status">Ready</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="model-card warn">'
            f'<img src="{_LUCIDE_CDN}/alert-triangle.svg" width="14" height="14" '
            f'style="filter:invert(1);opacity:0.4;flex-shrink:0;">'
            f'<div><div class="model-card-name">Not Found</div>'
            f'<div class="model-card-status">Run main.py</div></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="sb-footer">'
        f'<img src="{_LUCIDE_CDN}/graduation-cap.svg" width="11" height="11" '
        f'style="filter:invert(1);opacity:0.3;">'
        f'M.Tech Mini-Project &nbsp;·&nbsp; ML</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT TIP
# ══════════════════════════════════════════════════════════════════════════════
if page_id == "predict":
    st.markdown(pg_title("target", "Predict Tip"), unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Enter bill details. The selected model predicts the expected tip.</div>',
                unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown(sec_label("sliders-horizontal", "Bill Details"), unsafe_allow_html=True)
        total_bill = st.number_input(
            "Total Bill ($)",
            min_value=3.07, max_value=200.0, value=25.00, step=0.50,
            format="%.2f",
            help="Enter the total bill amount. Training data range: $3.07 – $50.81.",
        )
        size = st.slider("Party Size", min_value=1, max_value=6, value=2)
        c1, c2 = st.columns(2)
        with c1:
            # Pills for multi-option fields
            _sex = st.pills("Gender", ["Male", "Female"], default="Male", key="sex_pills")
            sex  = _sex if _sex is not None else "Male"

            _day = st.pills("Day", ["Thur", "Fri", "Sat", "Sun"], default="Sat", key="day_pills")
            day  = _day if _day is not None else "Sat"
        with c2:
            _smk   = st.pills("Smoker",    ["No", "Yes"],           default="No",     key="smoker_pills")
            smoker = _smk if _smk is not None else "No"

            _time  = st.pills("Meal Time", ["Lunch", "Dinner"],     default="Dinner", key="time_pills")
            time   = _time if _time is not None else "Dinner"
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    with col_result:
        st.markdown(sec_label("trending-up", "Result"), unsafe_allow_html=True)

        if predict_btn:
            if model is None:
                st.error("No model loaded. Run `python src/main.py` first.")
            else:
                features  = encode_features(total_bill, sex, smoker, day, time, size)
                predicted = max(0.5, model.predict(features)[0])
                tip_pct   = (predicted / total_bill) * 100
                total_amt = total_bill + predicted

                st.toast(f"Prediction complete — ${predicted:.2f} tip")
                st.markdown(f"""
                <div class="pred-hero">
                    <div class="pred-eyebrow">Predicted Tip &nbsp;·&nbsp; {selected_model_name}</div>
                    <div class="pred-num">${predicted:.2f}</div>
                    <div class="pred-pct">{tip_pct:.1f}% of bill &nbsp;·&nbsp; ${total_amt:.2f} total</div>
                    <div class="pred-rule"></div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Bill", f"${total_bill:.2f}")
                m2.metric("Tip",  f"${predicted:.2f}")
                m3.metric("Total", f"${total_amt:.2f}")

                industry = total_bill * 0.18
                diff = predicted - industry
                if diff >= 0:
                    st.markdown(
                        f'<div class="bench">Predicted tip is <strong>${diff:.2f} above</strong> '
                        f'the 18% industry standard (${industry:.2f}).</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="bench below">Predicted tip is <strong>${abs(diff):.2f} below</strong> '
                        f'the 18% industry standard (${industry:.2f}).</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div class="await-box"><p>Fill in the form and click Run Prediction</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(sec_label("list", "Batch Predictions — Sample Cases"), unsafe_allow_html=True)
    if model:
        rows = []
        for case in SAMPLE_CASES:
            feat = encode_features(**case)
            pred = max(0.5, model.predict(feat)[0])
            rows.append({**case,
                         'Predicted Tip ($)': round(pred, 2),
                         'Tip %': round((pred / case['total_bill']) * 100, 1)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Train the models first to see batch predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page_id == "explorer":
    st.markdown(pg_title("bar-chart-2", "Data Explorer"), unsafe_allow_html=True)
    _df_meta = load_data()
    _n_records = len(_df_meta) if _df_meta is not None else 0
    st.markdown(f'<div class="pg-sub">Explore the Tips dataset used for training — {_n_records:,} records, 7 features.</div>',
                unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("Dataset not found. Run `python src/download_data.py` first.")
        st.stop()

    st.markdown(sec_label("layout-dashboard", "Overview"), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Avg Bill", f"${df['total_bill'].mean():.2f}")
    c4.metric("Avg Tip",  f"${df['tip'].mean():.2f}")

    with st.expander("Raw Data"):
        st.dataframe(df, use_container_width=True)
    with st.expander("Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown(sec_label("pie-chart", "Distributions"), unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Histograms", "Categorical", "Correlation", "Scatter"])

    _plotly_layout = dict(
        template='plotly_dark',
        paper_bgcolor='#0B0B0B',
        plot_bgcolor='#1F2937',
        font=dict(family='DM Sans, sans-serif', color='#9CA3AF'),
        title_font=dict(color='#F9FAFB'),
        margin=dict(l=40, r=20, t=45, b=40),
    )

    with tab1:
        cols_hist = st.columns(3)
        for col_w, feat, clr in zip(cols_hist,
                                    ['total_bill', 'tip', 'size'],
                                    [_GOLD, _AMBER, '#a8872a']):
            fig_h = px.histogram(df, x=feat, nbins=25,
                                 title=feat.replace('_', ' ').title(),
                                 color_discrete_sequence=[clr])
            fig_h.update_layout(**_plotly_layout)
            col_w.plotly_chart(fig_h, use_container_width=True)

    with tab2:
        cols_box = st.columns(2)
        for idx, cat in enumerate(['sex', 'smoker', 'day', 'time']):
            fig_b = px.box(df, x=cat, y='tip', color=cat,
                           title=f'Tip by {cat.title()}',
                           color_discrete_sequence=px.colors.sequential.YlOrBr)
            fig_b.update_layout(**_plotly_layout, showlegend=False)
            cols_box[idx % 2].plotly_chart(fig_b, use_container_width=True)

    with tab3:
        numeric_df = df[['total_bill', 'tip', 'size']].copy()
        numeric_df['tip_pct'] = (df['tip'] / df['total_bill']) * 100
        corr_vals = numeric_df.corr().round(2)
        fig_c = px.imshow(corr_vals, text_auto=True, color_continuous_scale='RdYlGn',
                          zmin=-1, zmax=1, title='Feature Correlation')
        fig_c.update_layout(**_plotly_layout)
        st.plotly_chart(fig_c, use_container_width=True)

    with tab4:
        cols_sc = st.columns(2)
        for col_w, xcol in zip(cols_sc, ['total_bill', 'size']):
            fig_s = px.scatter(df, x=xcol, y='tip', opacity=0.5,
                               trendline='ols',
                               title=f'{xcol.replace("_", " ").title()} vs Tip',
                               color_discrete_sequence=[_MUTED],
                               trendline_color_override=_GOLD)
            fig_s.update_layout(**_plotly_layout)
            col_w.plotly_chart(fig_s, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page_id == "compare":
    st.markdown(pg_title("layers", "Model Comparison"), unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Performance of all 6 trained models evaluated on the held-out test set.</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading model metrics…"):
        try:
            metrics = get_model_metrics()
        except Exception as e:
            st.error(f"Could not compute metrics: {e}\nRun `python src/main.py` first.")
            st.stop()

    if not metrics:
        st.warning("No trained models found. Run `python src/main.py` first.")
        st.stop()

    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    metrics_df = metrics_df.sort_values('R² Score', ascending=False).reset_index(drop=True)
    metrics_df.insert(0, 'Rank', range(1, len(metrics_df) + 1))

    st.markdown(sec_label("table", "Performance Table"), unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style
            .highlight_max(subset=['R² Score'], color='#2a2210')
            .highlight_min(subset=['RMSE ($)', 'MAE ($)'], color='#2a2210')
            .format({'R² Score': '{:.4f}', 'RMSE ($)': '{:.4f}', 'MAE ($)': '{:.4f}'}),
        use_container_width=True,
    )

    st.markdown(sec_label("bar-chart-2", "Visual Comparison"), unsafe_allow_html=True)
    tab_r2, tab_rmse, tab_mae = st.tabs(["R² Score", "RMSE", "MAE"])

    _cmp_layout = dict(
        template='plotly_dark', paper_bgcolor='#0B0B0B', plot_bgcolor='#1F2937',
        font=dict(family='DM Sans, sans-serif', color='#9CA3AF'),
        title_font=dict(color='#F9FAFB'),
        margin=dict(l=40, r=20, t=45, b=100),
        xaxis_tickangle=-25,
    )

    with tab_r2:
        fig_r2 = px.bar(metrics_df, x='Model', y='R² Score', text='R² Score',
                        title='R² Score by Model  (higher = better)',
                        color='R² Score', color_continuous_scale='YlOrBr')
        fig_r2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_r2.update_layout(**_cmp_layout)
        st.plotly_chart(fig_r2, use_container_width=True)

    with tab_rmse:
        rmse_s = metrics_df.sort_values('RMSE ($)')
        fig_rm = px.bar(rmse_s, x='Model', y='RMSE ($)', text='RMSE ($)',
                        title='RMSE by Model  (lower = better)',
                        color='RMSE ($)', color_continuous_scale='YlOrBr_r')
        fig_rm.update_traces(texttemplate='$%{text:.4f}', textposition='outside')
        fig_rm.update_layout(**_cmp_layout)
        st.plotly_chart(fig_rm, use_container_width=True)

    with tab_mae:
        mae_s = metrics_df.sort_values('MAE ($)')
        fig_ma = px.bar(mae_s, x='Model', y='MAE ($)', text='MAE ($)',
                        title='MAE by Model  (lower = better)',
                        color='MAE ($)', color_continuous_scale='YlOrBr_r')
        fig_ma.update_traces(texttemplate='$%{text:.4f}', textposition='outside')
        fig_ma.update_layout(**_cmp_layout)
        st.plotly_chart(fig_ma, use_container_width=True)

    best = metrics_df.iloc[0]
    st.markdown(
        f'<div class="best-strip">'
        f'{lucide("trophy", size=16, css_class="li-muted")}'
        f'<span>Best Model</span>{best["Model"]}'
        f'<span class="dim">R² {best["R² Score"]:.4f}</span>'
        f'<span class="dim">RMSE ${best["RMSE ($)"]:.4f}</span>'
        f'<span class="dim">MAE ${best["MAE ($)"]:.4f}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page_id == "about":
    st.markdown(pg_title("info", "About"), unsafe_allow_html=True)
    st.markdown(
        "<div class='pg-sub'>Waiter's Tips Prediction System — M.Tech Mini-Project in Machine Learning.</div>",
        unsafe_allow_html=True,
    )

    _about_df = load_data()
    _about_n  = len(_about_df) if _about_df is not None else 0
    st.markdown(f"""
<div class="stat-grid">
    <div class="stat-cell"><div class="stat-lbl">Dataset Records</div><div class="stat-val">{_about_n:,}</div></div>
    <div class="stat-cell"><div class="stat-lbl">Models Trained</div><div class="stat-val">3</div></div>
    <div class="stat-cell"><div class="stat-lbl">Input Features</div><div class="stat-val">6</div></div>
    <div class="stat-cell"><div class="stat-lbl">Algorithm Family</div><div class="stat-val">Trees</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown(sec_label("file-text", "What This System Does"), unsafe_allow_html=True)
    st.markdown("""
Predicts restaurant tip amounts based on bill details using three decision-tree-based
ML models trained on 1,000 synthetic records that mirror real restaurant data.

**Input features:** total bill · party size · day · meal time · gender · smoker status
    """)

    st.markdown(sec_label("check-square", "Models Implemented"), unsafe_allow_html=True)
    st.markdown("""
| # | Model | Algorithm | Criterion |
|---|-------|-----------|-----------|
| 1 | **ID3 Tree** | Decision Tree | Friedman MSE (information-gain style) |
| 2 | **CART Tree** | Decision Tree | Squared Error (standard CART) |
| 3 | **Random Forest** | Ensemble of 200 CART trees | Squared Error |
    """)

    st.markdown(sec_label("box", "Models"), unsafe_allow_html=True)
    st.markdown(
        '<span class="pill">ID3 Tree</span>'
        '<span class="pill">CART Tree</span>'
        '<span class="pill">Random Forest</span>',
        unsafe_allow_html=True,
    )

    st.markdown(sec_label("code-2", "Tech Stack"), unsafe_allow_html=True)
    st.markdown(
        '<span class="pill">Python</span><span class="pill">scikit-learn</span>'
        '<span class="pill">Streamlit</span><span class="pill">pandas</span>'
        '<span class="pill">matplotlib</span><span class="pill">seaborn</span>',
        unsafe_allow_html=True,
    )

    _ibox_df = load_data()
    _ibox_n  = len(_ibox_df) if _ibox_df is not None else 0
    st.markdown(
        f'<div class="ibox" style="margin-top:2rem">Dataset: Synthetic Tips Dataset — '
        f'{_ibox_n:,} records · 6 features · no missing values.</div>',
        unsafe_allow_html=True,
    )
