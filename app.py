"""
================================================================================
  DASHBOARD INTERACTIVO DE ASESORÍA DE INVERSIÓN — FICs Colombia 2026
  Estilo Fintech · Streamlit · Procesamiento 100% en memoria
================================================================================
  Ejecutar con:  streamlit run app.py
  Python 3.9+  |  Ver requirements.txt para dependencias
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import io
import math
import base64
from datetime import datetime

import numpy  as np
import pandas as pd
import plotly.express     as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN DE PÁGINA (debe ser el primer comando Streamlit)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "FIC Advisor 2026",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. ESTILOS GLOBALES (Dark-mode ejecutivo + acento fintech)
# ──────────────────────────────────────────────────────────────────────────────
COLORES = {
    "bg"         : "#0F1117",
    "surface"    : "#1A1D27",
    "surface2"   : "#22263A",
    "primary"    : "#4F8EF7",
    "accent"     : "#00D4A0",
    "danger"     : "#FF4E6A",
    "warning"    : "#FFAA40",
    "text"       : "#E8ECF4",
    "muted"      : "#8892A4",
    "perfil1"    : "#FF4E6A",   # Riesgo Elevado
    "perfil2"    : "#00D4A0",   # Sólidos / Estables
    "perfil3"    : "#4F8EF7",   # Mega Fondos
}

CLUSTER_META = {
    # Se asignan dinámicamente tras el clustering; aquí la plantilla
    "P1": {"label": "Perfil 1 · Riesgo Elevado",        "emoji": "🔴", "color": COLORES["perfil1"],
           "desc": "Rentabilidad negativa o muy baja con alta volatilidad. Señal de alerta."},
    "P2": {"label": "Perfil 2 · Fondos Sólidos",         "emoji": "🟢", "color": COLORES["perfil2"],
           "desc": "Rentabilidad positiva y volatilidad controlada. Recomendación principal."},
    "P3": {"label": "Perfil 3 · Mega Fondos Institucionales", "emoji": "🔵", "color": COLORES["perfil3"],
           "desc": "Patrimonio masivo, alta liquidez y eficiencia institucional."},
}

st.markdown(f"""
<style>
  /* ── Base ── */
  .stApp {{ background-color: {COLORES["bg"]}; color: {COLORES["text"]}; }}
  section[data-testid="stSidebar"] {{ background-color: {COLORES["surface"]}; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }}

  /* ── KPI cards ── */
  .kpi-card {{
    background: {COLORES["surface"]}; border-radius: 14px;
    padding: 22px 24px; border-left: 4px solid {COLORES["primary"]};
    box-shadow: 0 4px 20px rgba(0,0,0,.35);
  }}
  .kpi-label  {{ font-size: 11px; font-weight: 700; letter-spacing: 1px;
                 text-transform: uppercase; color: {COLORES["muted"]}; margin-bottom: 6px; }}
  .kpi-value  {{ font-size: 28px; font-weight: 800; color: {COLORES["text"]}; line-height: 1.1; }}
  .kpi-sub    {{ font-size: 12px; color: {COLORES["muted"]}; margin-top: 4px; }}

  /* ── Cluster cards ── */
  .cluster-card {{
    background: {COLORES["surface2"]}; border-radius: 14px;
    padding: 20px 22px; height: 100%;
    box-shadow: 0 4px 20px rgba(0,0,0,.25);
  }}
  .cluster-title {{ font-size: 15px; font-weight: 700; margin-bottom: 10px; }}
  .cluster-stat  {{ font-size: 13px; color: {COLORES["muted"]}; margin-bottom: 4px; }}

  /* ── Section headers ── */
  .section-hdr {{
    font-size: 14px; font-weight: 700; color: {COLORES["muted"]};
    text-transform: uppercase; letter-spacing: 1.2px;
    border-bottom: 1px solid {COLORES["surface2"]}; padding-bottom: 8px;
    margin-bottom: 18px;
  }}

  /* ── Welcome screen ── */
  .welcome-box {{
    background: linear-gradient(135deg, {COLORES["surface"]} 0%, {COLORES["surface2"]} 100%);
    border-radius: 20px; padding: 48px 52px;
    box-shadow: 0 8px 40px rgba(0,0,0,.5);
    border: 1px solid rgba(79,142,247,.2);
    text-align: center;
  }}
  .welcome-title {{ font-size: 36px; font-weight: 800; color: {COLORES["text"]}; }}
  .welcome-sub   {{ font-size: 16px; color: {COLORES["muted"]}; margin-top: 10px; }}
  .tag {{ display: inline-block; background: rgba(79,142,247,.15);
          color: {COLORES["primary"]}; border-radius: 99px;
          padding: 4px 14px; font-size: 12px; font-weight: 600;
          margin: 4px 3px; }}

  /* ── Rec cards ── */
  .rec-card {{
    background: {COLORES["surface2"]}; border-radius: 12px;
    padding: 18px 20px; border-top: 3px solid {COLORES["accent"]};
    margin-bottom: 12px;
  }}
  .rec-title  {{ font-size: 13px; font-weight: 700; color: {COLORES["text"]}; }}
  .rec-ent    {{ font-size: 11px; color: {COLORES["muted"]}; margin-top: 2px; }}
  .rec-metric {{ font-size: 12px; color: {COLORES["accent"]}; font-weight: 600; }}

  /* ── Progress / steps ── */
  .step-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: {COLORES["primary"]}; color: #fff;
    font-size: 13px; font-weight: 700; margin-right: 10px;
  }}
  .step-label {{ font-size: 14px; font-weight: 600; color: {COLORES["text"]}; }}

  /* ── Plotly overrides ── */
  .js-plotly-plot .plotly .modebar {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. HELPERS DE FORMATO
# ──────────────────────────────────────────────────────────────────────────────
def fmt_cop(v: float, decimals=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    if abs(v) >= 1e12: return f"${v/1e12:,.{decimals}f} Bill."
    if abs(v) >= 1e9:  return f"${v/1e9:,.{decimals}f} MM"
    if abs(v) >= 1e6:  return f"${v/1e6:,.{decimals}f} M"
    return f"${v:,.0f}"

def fmt_pct(v: float, sign=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    s = "+" if sign and v > 0 else ""
    return f"{s}{v:.4f}%"

def fmt_num(v, decimals=0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return f"{v:,.{decimals}f}"

def kpi_card(label: str, value: str, sub: str = "", color: str = None) -> str:
    border = color or COLORES["primary"]
    return f"""
    <div class="kpi-card" style="border-left-color:{border}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""


# ──────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE DE DATOS (cached — solo corre una vez por sesión)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=0)
def cargar_y_procesar(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Ejecuta el pipeline completo:
      1. Carga y limpieza
      2. Filtrado (fondos abiertos, activos, sin nulos)
      3. Agregación por ID_FONDO_SERIE
      4. Clustering K-Means (k=3)
    Devuelve: (df_detalle_limpio, df_resumen_clustered, meta_dict)
    """

    # ── 3.1 Carga — detecta separador (coma o tabulación) y encoding ─────────
    def _read(enc: str) -> pd.DataFrame:
        """Detecta automáticamente si el separador es tabulación o coma."""
        sample = file_bytes[:4096].decode(enc, errors="replace")
        # Si hay más tabs que comas en el encabezado → TSV
        first_line = sample.split("\n")[0]
        sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
        return pd.read_csv(
            io.BytesIO(file_bytes),
            sep       = sep,
            encoding  = enc,
            low_memory= False,
            na_values = ["", "NA", "N/A", "NULL"],
        )

    try:
        raw = _read("utf-8-sig")
    except UnicodeDecodeError:
        raw = _read("latin-1")

    # Normalizar nombres de columna
    raw.columns = [
        c.strip()
         .upper()
         .replace(" ", "_")
         .encode("ascii", "ignore").decode()   # quitar tildes en nombres col.
        for c in raw.columns
    ]
    # BOM fix en primera columna
    raw.columns = [raw.columns[0].lstrip("\ufeff")] + list(raw.columns[1:])

    # Columnas numéricas a limpiar (comas de miles)
    COLS_NUM = [
        "RENDIMIENTOS_ABONADOS", "PRECIERRE_FONDO_DIA_T",
        "NUMERO_UNIDADES_FONDO_CIERRE_OPER_DIA_T_ANTERIOR",
        "VALOR_UNIDAD_OPERACIONES_DIA_T", "APORTES_RECIBIDOS",
        "RETIROS_REDENCIONES", "ANULACIONES", "VALOR_FONDO_CIERRE_DIA_T",
        "NUMERO_INVERSIONISTAS", "RENTABILIDAD_DIARIA",
        "RENTABILIDAD_MENSUAL", "RENTABILIDAD_SEMESTRAL", "RENTABILIDAD_ANUAL",
    ]
    for col in COLS_NUM:
        if col in raw.columns:
            raw[col] = (
                raw[col].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace('"', "", regex=False)
                        .pipe(pd.to_numeric, errors="coerce")
            )

    # Fecha
    if "FECHA_CORTE" in raw.columns:
        raw["FECHA_CORTE"] = pd.to_datetime(
            raw["FECHA_CORTE"].astype(str).str.strip(),
            dayfirst=True, errors="coerce",
        )

    # ── 3.2 Limpieza de texto ─────────────────────────────────────────────────
    for col in ["NOMBRE_ENTIDAD", "NOMBRE_PATRIMONIO",
                "NOMBRE_SUBTIPO_PATRIMONIO", "TIPO_PARTICIPACION"]:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.replace('"', "").str.strip()

    # ── 3.3 Filtrado (spec exacta del R) ────────────────────────────────────
    df = raw.copy()
    df = df[df["VALOR_FONDO_CIERRE_DIA_T"] > 0]
    df = df[df["NUMERO_INVERSIONISTAS"] > 0]

    # Solo fondos abiertos
    nom = df["NOMBRE_PATRIMONIO"].str.upper() if "NOMBRE_PATRIMONIO" in df.columns else pd.Series(dtype=str)
    sub = df["NOMBRE_SUBTIPO_PATRIMONIO"].str.upper() if "NOMBRE_SUBTIPO_PATRIMONIO" in df.columns else pd.Series(dtype=str)

    mask_abierto = (
        nom.str.contains("ABIERTO", na=False) |
        sub.str.contains("ABIERTO|GENERAL", na=False)
    ) & ~nom.str.contains("CERRADO", na=False)
    df = df[mask_abierto]

    # Limpiar outliers severos de rentabilidad diaria
    df = df[df["RENTABILIDAD_DIARIA"].between(-100, 100)]
    df = df.dropna(subset=["RENTABILIDAD_DIARIA", "VALOR_FONDO_CIERRE_DIA_T"])

    # Flujo neto e ID de serie
    df["FLUJO_NETO"] = df["APORTES_RECIBIDOS"].fillna(0) - df["RETIROS_REDENCIONES"].fillna(0)

    codigo_col = "CODIGO_NEGOCIO" if "CODIGO_NEGOCIO" in df.columns else None
    if codigo_col:
        df["CODIGO_NEGOCIO_CLEAN"] = df[codigo_col].astype(str).str.replace(",", "", regex=False).str.strip()
        df["ID_FONDO_SERIE"] = df["CODIGO_NEGOCIO_CLEAN"] + "_" + df["TIPO_PARTICIPACION"].astype(str)
    else:
        df["ID_FONDO_SERIE"] = df["NOMBRE_PATRIMONIO"] + "_" + df["TIPO_PARTICIPACION"].astype(str)

    # ── 3.4 Agregación por fondo ──────────────────────────────────────────────
    # NOTA: En el CSV de la Superfinanciera, RENTABILIDAD_DIARIA contiene
    # la rentabilidad efectiva anual (valores ~7–12%), NO una tasa diaria.
    # Usamos RENTABILIDAD_ANUAL directamente para evitar anualizaciones erróneas.
    grp = df.groupby(
        ["ID_FONDO_SERIE", "NOMBRE_ENTIDAD", "NOMBRE_PATRIMONIO"],
        as_index=False
    ).agg(
        DIAS_OBSERVADOS        = pd.NamedAgg(column="RENTABILIDAD_ANUAL",  aggfunc="count"),
        PATRIMONIO_MEDIO       = pd.NamedAgg(column="VALOR_FONDO_CIERRE_DIA_T", aggfunc="mean"),
        INVERSIONISTAS_MEDIO   = pd.NamedAgg(column="NUMERO_INVERSIONISTAS",    aggfunc="mean"),
        RENTABILIDAD_MEDIA     = pd.NamedAgg(column="RENTABILIDAD_ANUAL",        aggfunc="mean"),
        VOLATILIDAD_DIARIA     = pd.NamedAgg(column="RENTABILIDAD_ANUAL",        aggfunc="std"),
        FLUJO_NETO_TOTAL       = pd.NamedAgg(column="FLUJO_NETO",                aggfunc="sum"),
        RENTABILIDAD_ANUAL_REF = pd.NamedAgg(column="RENTABILIDAD_ANUAL",        aggfunc="mean"),
        RENTABILIDAD_MENSUAL   = pd.NamedAgg(column="RENTABILIDAD_MENSUAL",      aggfunc="mean")
    )

    grp = grp[grp["DIAS_OBSERVADOS"] >= 5]
    grp = grp.dropna(subset=["VOLATILIDAD_DIARIA"])
    grp = grp[grp["VOLATILIDAD_DIARIA"] > 0]

    # RENTABILIDAD_MEDIA ya es % EA — no aplicar compuesto.
    # VOLATILIDAD_DIARIA aquí es la std de las rentabilidades anuales (proxy de dispersión).
    grp["RENTABILIDAD_ANUAL_EA"]  = grp["RENTABILIDAD_MEDIA"]          # ya es EA
    grp["VOLATILIDAD_ANUAL"]      = grp["VOLATILIDAD_DIARIA"]          # std de EA cross-series
    grp["RATIO_RETORNO_RIESGO"]   = np.where(
        grp["VOLATILIDAD_ANUAL"] > 0,
        grp["RENTABILIDAD_ANUAL_EA"] / grp["VOLATILIDAD_ANUAL"],
        0.0,
    )
    grp["LOG_PATRIMONIO"]         = np.log10(grp["PATRIMONIO_MEDIO"].clip(lower=1))
    grp["FLUJO_NETO_RELATIVO"]    = grp["FLUJO_NETO_TOTAL"] / grp["PATRIMONIO_MEDIO"].clip(lower=1)

    # ── 3.5 K-Means (k=3) — features sin data leakage ────────────────────────
    FEATURES = ["RENTABILIDAD_ANUAL_EA", "VOLATILIDAD_ANUAL",
                "RATIO_RETORNO_RIESGO",  "LOG_PATRIMONIO"]
    X = grp[FEATURES].fillna(0).values
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, n_init=25, random_state=123)
    km.fit(X_sc)
    grp["CLUSTER_RAW"] = km.labels_

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=FEATURES,
    )
    # Perfil 2 = mayor RENTABILIDAD_ANUAL_EA y menor VOLATILIDAD → sólidos
    # Perfil 1 = menor RENTABILIDAD_ANUAL_EA (o negativa) → riesgo elevado
    # Perfil 3 = mayor LOG_PATRIMONIO → mega fondos
    cent_idx = {
        "P2": int(centroids["RENTABILIDAD_ANUAL_EA"].idxmax()),
        "P3": int(centroids["LOG_PATRIMONIO"].idxmax()),
    }
    used = set(cent_idx.values())
    cent_idx["P1"] = [i for i in range(3) if i not in used][0]

    label_map = {v: k for k, v in cent_idx.items()}
    grp["PERFIL"] = grp["CLUSTER_RAW"].map(label_map)

    # ── Meta del clustering ───────────────────────────────────────────────────
    # RENTABILIDAD_ANUAL_EA ya está calculada correctamente en grp:
    #   grp["RENTABILIDAD_ANUAL_EA"] = ((1 + RENTABILIDAD_MEDIA/100)^252 - 1) * 100
    # Usarla directamente evita aplicar el compuesto dos veces.
    rent_prom = float(np.average(
        grp["RENTABILIDAD_ANUAL_EA"],
        weights=grp["PATRIMONIO_MEDIO"],
    ))

    meta = {
        "n_fondos"      : len(grp),
        "n_entidades"   : grp["NOMBRE_ENTIDAD"].nunique(),
        "fecha_min"     : df["FECHA_CORTE"].min() if "FECHA_CORTE" in df.columns else None,
        "fecha_max"     : df["FECHA_CORTE"].max() if "FECHA_CORTE" in df.columns else None,
        "patrimonio_tot": grp["PATRIMONIO_MEDIO"].sum(),
        "rent_prom"     : rent_prom,
        "vol_prom"      : grp["VOLATILIDAD_ANUAL"].mean(),
        "centroids"     : centroids.to_dict(),
        "perfil_counts" : grp["PERFIL"].value_counts().to_dict(),
        "filename"      : filename,
    }
    return df, grp, meta


# ──────────────────────────────────────────────────────────────────────────────
# 4. PANTALLA DE BIENVENIDA
# ──────────────────────────────────────────────────────────────────────────────
def pantalla_bienvenida():
    st.markdown("""
    <div class="welcome-box">
      <div style="font-size:52px;margin-bottom:10px">📊</div>
      <div class="welcome-title">FIC Advisor 2026</div>
      <div class="welcome-sub">Dashboard de Asesoría de Inversión · Fondos de Inversión Colectiva Colombia</div>
      <div style="margin-top:20px">
        <span class="tag">🤖 K-Means Clustering</span>
        <span class="tag">📈 Análisis Riesgo/Retorno</span>
        <span class="tag">🎯 Recomendador Personalizado</span>
        <span class="tag">100% en Memoria</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="section-hdr">PASO 1 · CARGAR DATOS</div>', unsafe_allow_html=True)
        archivo = st.file_uploader(
            label       = "Selecciona el CSV oficial de la Superfinanciera",
            type        = ["csv"],
            help        = "Archivo: Rentabilidades_de_los_Fondos_de_Inversión_Colectiva_(FIC)_*.csv",
            label_visibility = "collapsed",
        )
        st.caption("📁 Archivo esperado: `Rentabilidades_de_los_Fondos_de_Inversión_Colectiva_(FIC)_*.csv`")
        return archivo


# ──────────────────────────────────────────────────────────────────────────────
# 5. PESTAÑA 1 — VISIÓN GENERAL DEL MERCADO
# ──────────────────────────────────────────────────────────────────────────────
def tab_vision_general(df_det: pd.DataFrame, grp: pd.DataFrame, meta: dict):

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📌 MÉTRICAS CLAVE DEL MERCADO FIC</div>',
                unsafe_allow_html=True)

    # meta["rent_prom"] = promedio ponderado por patrimonio de RENTABILIDAD_ANUAL_EA (% EA)
    # Calculado en cargar_y_procesar() — no reaplicar compuesto aquí.
    rent_ea = meta["rent_prom"]

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "FONDOS ANALIZADOS",  fmt_num(meta["n_fondos"]),
             f"{meta['n_entidades']} entidades fiduciarias",    COLORES["primary"]),
        (c2, "PATRIMONIO TOTAL",   fmt_cop(meta["patrimonio_tot"]),
             "Suma de patrimonio medio por fondo",               COLORES["accent"]),
        (c3, "RENTAB. PROMEDIO EA", f"{rent_ea:+.2f}%",
             "Rentabilidad Efectiva Anual pond. por patrimonio",
             COLORES["accent"] if rent_ea >= 0 else COLORES["danger"]),
        (c4, "VOLATILIDAD ANUAL",  f"{meta['vol_prom']:.2f}%",
             "Volatilidad anualizada promedio (σ × √252)",       COLORES["warning"]),
        (c5, "PERÍODO ANALIZADO",
             f"{meta['fecha_max'].strftime('%b %Y') if meta['fecha_max'] else '—'}",
             (f"{meta['fecha_min'].strftime('%d %b %Y')} → {meta['fecha_max'].strftime('%d %b %Y')}"
              if meta["fecha_min"] else ""),                     COLORES["muted"]),
    ]
    for col, lbl, val, sub, clr in cards:
        with col:
            st.markdown(kpi_card(lbl, val, sub, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter Riesgo vs Retorno ─────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-hdr">🎯 RIESGO vs. RETORNO</div>',
                    unsafe_allow_html=True)

        color_map = {"P1": COLORES["perfil1"], "P2": COLORES["perfil2"], "P3": COLORES["perfil3"]}
        label_map = {"P1": CLUSTER_META["P1"]["label"],
                     "P2": CLUSTER_META["P2"]["label"],
                     "P3": CLUSTER_META["P3"]["label"]}

        plot_df = grp.copy()
        plot_df["PERFIL_LABEL"] = plot_df["PERFIL"].map(label_map)
        plot_df["PAT_NORM"]     = np.clip(plot_df["PATRIMONIO_MEDIO"] / plot_df["PATRIMONIO_MEDIO"].max() * 60 + 4, 4, 64)
        plot_df["NOMBRE_CORTO"] = plot_df["NOMBRE_PATRIMONIO"].str[:40]

    fig_scatter = px.scatter(
            plot_df,
            x            = "VOLATILIDAD_ANUAL",             # <- Cambiado a anual
            y            = "RENTABILIDAD_ANUAL_EA",         # <- Cambiado a EA
            size         = "PAT_NORM",
            color        = "PERFIL_LABEL",
            color_discrete_map = {v: color_map[k] for k, v in label_map.items()},
            hover_name   = "NOMBRE_CORTO",
            hover_data   = {
                "NOMBRE_ENTIDAD"       : True,
                "RATIO_RETORNO_RIESGO" : ":.4f",
                "PATRIMONIO_MEDIO"     : ":,.0f",
                "VOLATILIDAD_ANUAL"    : ":.2f",
                "RENTABILIDAD_ANUAL_EA": ":.2f",
                "PAT_NORM"             : False,
                "PERFIL_LABEL"         : False,
            },
            labels = {
                "VOLATILIDAD_ANUAL"     : "Volatilidad Anual (%)",
                "RENTABILIDAD_ANUAL_EA" : "Rentabilidad Efectiva Anual (EA %)",
                "PERFIL_LABEL"          : "Perfil",
            },
            title = "Relación Riesgo vs. Retorno — Universo FIC Colombia",
        )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,.3)",
                              annotation_text="Rentabilidad = 0",
                              annotation_position="bottom right")
    fig_scatter.update_layout(
            plot_bgcolor  = COLORES["surface"],
            paper_bgcolor = COLORES["surface"],
            font_color    = COLORES["text"],
            legend        = dict(orientation="h", yanchor="bottom", y=1.02,
                                 xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            margin        = dict(l=10, r=10, t=50, b=10),
            height        = 440,
        )
    fig_scatter.update_xaxes(gridcolor=COLORES["surface2"], zerolinecolor=COLORES["surface2"])
    fig_scatter.update_yaxes(gridcolor=COLORES["surface2"], zerolinecolor=COLORES["surface2"])
    st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-hdr">📊 DISTRIBUCIÓN POR PERFIL</div>',
                    unsafe_allow_html=True)

        perfil_counts = grp["PERFIL"].value_counts().reset_index()
        perfil_counts.columns = ["Perfil", "Fondos"]
        perfil_counts["Label"] = perfil_counts["Perfil"].map(label_map)

        fig_pie = px.pie(
            perfil_counts,
            names  = "Label",
            values = "Fondos",
            color  = "Label",
            color_discrete_map = {v: color_map[k] for k, v in label_map.items()},
            hole   = 0.55,
        )
        fig_pie.update_traces(
            textposition = "outside",
            textinfo     = "percent+label",
            textfont     = dict(size=11),
            marker       = dict(line=dict(color=COLORES["bg"], width=2)),
        )
        fig_pie.update_layout(
            plot_bgcolor  = COLORES["surface"],
            paper_bgcolor = COLORES["surface"],
            font_color    = COLORES["text"],
            showlegend    = False,
            margin        = dict(l=0, r=0, t=20, b=0),
            height        = 260,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Mini stats por perfil
# Mini stats por perfil (Actualizado a EA)
        for perfil_key in ["P2", "P3", "P1"]:
            sub_df = grp[grp["PERFIL"] == perfil_key]
            if sub_df.empty: continue
            meta_p = CLUSTER_META[perfil_key]
            st.markdown(f"""
            <div style="background:{COLORES['surface2']};border-radius:8px;
                        padding:10px 14px;margin-bottom:8px;
                        border-left:3px solid {meta_p['color']}">
              <span style="color:{meta_p['color']};font-weight:700;font-size:12px">
                {meta_p['emoji']} {meta_p['label']}
              </span>
              <span style="color:{COLORES['muted']};font-size:11px;float:right">
                {len(sub_df)} fondos
              </span><br>
              <span style="color:{COLORES['muted']};font-size:11px">
                Rent. Anual EA: {fmt_pct(sub_df['RENTABILIDAD_ANUAL_EA'].mean())} &nbsp;|&nbsp;
                Vol Anual: {fmt_pct(sub_df['VOLATILIDAD_ANUAL'].mean())}
              </span>
            </div>""", unsafe_allow_html=True)

    # ── Heatmap correlación ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🔗 MATRIZ DE CORRELACIÓN</div>',
                unsafe_allow_html=True)

    COLS_CORR = {
        "LOG_PATRIMONIO"       : "Log Patrimonio",
        "INVERSIONISTAS_MEDIO" : "Inversionistas",
        "RENTABILIDAD_MEDIA"   : "Rentabilidad",
        "VOLATILIDAD_DIARIA"   : "Volatilidad",
        "RATIO_RETORNO_RIESGO" : "Retorno/Riesgo",
        "FLUJO_NETO_RELATIVO"  : "Flujo Neto Rel.",
    }
    corr_df = grp[[c for c in COLS_CORR if c in grp.columns]].copy()
    corr_df.columns = [COLS_CORR[c] for c in corr_df.columns]
    corr = corr_df.corr(numeric_only=True)

    fig_heat = go.Figure(go.Heatmap(
        z           = corr.values,
        x           = corr.columns.tolist(),
        y           = corr.index.tolist(),
        colorscale  = [[0, COLORES["danger"]], [0.5, COLORES["surface2"]], [1, COLORES["accent"]]],
        zmin=-1, zmax=1,
        text        = np.round(corr.values, 2),
        texttemplate= "%{text}",
        textfont    = dict(size=12, color=COLORES["text"]),
        hoverongaps = False,
    ))
    fig_heat.update_layout(
        plot_bgcolor  = COLORES["surface"],
        paper_bgcolor = COLORES["surface"],
        font_color    = COLORES["text"],
        margin        = dict(l=10, r=10, t=20, b=10),
        height        = 360,
        xaxis         = dict(tickangle=-35),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("ℹ️  ¿Qué mide cada variable?"):
        st.markdown("""
| Variable | Descripción |
|---|---|
| **Log Patrimonio** | Tamaño del fondo en escala logarítmica. Suaviza el efecto de mega-fondos. |
| **Inversionistas** | Promedio de inversionistas durante el período. Proxy de demanda. |
| **Rentabilidad** | Rentabilidad diaria media. Retorno esperado del fondo. |
| **Volatilidad** | Desviación estándar de la rentabilidad diaria. Proxy del riesgo. |
| **Retorno/Riesgo** | Ratio de Sharpe simplificado (Rf = 0). Mayor es mejor. |
| **Flujo Neto Rel.** | (Aportes - Retiros) / Patrimonio. Señal de captación o desinversión. |
        """)


# ──────────────────────────────────────────────────────────────────────────────
# 6. PESTAÑA 2 — EXPLORADOR DE CLUSTERS
# ──────────────────────────────────────────────────────────────────────────────
def tab_explorador(grp: pd.DataFrame):

    # ── Filtros ───────────────────────────────────────────────────────────────
    with st.container():
        f1, f2, f3 = st.columns([2, 2, 2])
        entidades  = ["Todas"] + sorted(grp["NOMBRE_ENTIDAD"].dropna().unique().tolist())
        perfiles   = ["Todos"] + [f"{CLUSTER_META[k]['emoji']} {CLUSTER_META[k]['label']}"
                                   for k in ["P1", "P2", "P3"]]
        tam_opts   = ["Todos", "Grande (>$100MM)", "Mediano ($10–100MM)", "Pequeño (<$10MM)"]

        sel_ent   = f1.selectbox("🏦 Entidad / Fiduciaria", entidades)
        sel_perf  = f2.selectbox("🎯 Perfil de Cluster",    perfiles)
        sel_tam   = f3.selectbox("💰 Tamaño del Fondo",     tam_opts)

    df_f = grp.copy()

    if sel_ent != "Todas":
        df_f = df_f[df_f["NOMBRE_ENTIDAD"] == sel_ent]

    if sel_perf != "Todos":
        pkey = next(k for k in ["P1","P2","P3"]
                    if CLUSTER_META[k]["label"] in sel_perf)
        df_f = df_f[df_f["PERFIL"] == pkey]

    if sel_tam == "Grande (>$100MM)":
        df_f = df_f[df_f["PATRIMONIO_MEDIO"] >  1e11]
    elif sel_tam == "Mediano ($10–100MM)":
        df_f = df_f[(df_f["PATRIMONIO_MEDIO"] >= 1e10) & (df_f["PATRIMONIO_MEDIO"] <= 1e11)]
    elif sel_tam == "Pequeño (<$10MM)":
        df_f = df_f[df_f["PATRIMONIO_MEDIO"] <  1e10]

    st.caption(f"Mostrando **{len(df_f):,}** fondos de **{len(grp):,}** totales")

    # ── Tarjetas de Perfil ────────────────────────────────────────────────────
# ── Tarjetas de Perfil ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🗂️  PERFILES DE CLUSTER</div>',
                unsafe_allow_html=True)

    cols_pc = st.columns(3)
    for i, pkey in enumerate(["P2", "P3", "P1"]):
        sub = grp[grp["PERFIL"] == pkey]
        if sub.empty: continue
        meta_p = CLUSTER_META[pkey]
        with cols_pc[i]:
            st.markdown(f"""
            <div class="cluster-card" style="border-top:3px solid {meta_p['color']}">
              <div class="cluster-title" style="color:{meta_p['color']}">
                {meta_p['emoji']} {meta_p['label']}
              </div>
              <div class="cluster-stat">Fondos: <strong>{len(sub):,}</strong></div>
              <div class="cluster-stat">Patrimonio Total: <strong>{fmt_cop(sub['PATRIMONIO_MEDIO'].sum())}</strong></div>
              <div class="cluster-stat">Rentabilidad Anual EA: <strong>{fmt_pct(sub['RENTABILIDAD_ANUAL_EA'].mean())}</strong></div>
              <div class="cluster-stat">Volatilidad Anual: <strong>{fmt_pct(sub['VOLATILIDAD_ANUAL'].mean())}</strong></div>
              <div class="cluster-stat">Ratio Retorno/Riesgo: <strong>{sub['RATIO_RETORNO_RIESGO'].mean():.4f}</strong></div>
              <div style="margin-top:10px;font-size:11.5px;color:{COLORES['muted']}">
                {meta_p['desc']}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico de barras por entidad ─────────────────────────────────────────
    st.markdown('<div class="section-hdr">🏦  RENTABILIDAD MEDIA POR ENTIDAD</div>',
                unsafe_allow_html=True)

    ent_grp = (
        df_f.groupby("NOMBRE_ENTIDAD")
            .agg(Rent=("RENTABILIDAD_MEDIA","mean"),
                 Fondos=("ID_FONDO_SERIE","count"))
            .sort_values("Rent", ascending=False)
            .head(20)
            .reset_index()
    )
    ent_grp["Color"] = ent_grp["Rent"].apply(
        lambda r: COLORES["accent"] if r >= 0 else COLORES["danger"]
    )
    fig_bar = go.Figure(go.Bar(
        x         = ent_grp["Rent"],
        y         = ent_grp["NOMBRE_ENTIDAD"].str[:30],
        orientation = "h",
        marker_color= ent_grp["Color"],
        text        = ent_grp["Rent"].apply(lambda r: fmt_pct(r)),
        textposition= "outside",
        customdata  = ent_grp[["Fondos"]],
        hovertemplate = "<b>%{y}</b><br>Rent.: %{x:.4f}%<br>Fondos: %{customdata[0]}<extra></extra>",
    ))
    fig_bar.add_vline(x=0, line_color="rgba(255,255,255,.3)", line_dash="dash")
    fig_bar.update_layout(
        plot_bgcolor=COLORES["surface"], paper_bgcolor=COLORES["surface"],
        font_color=COLORES["text"], showlegend=False,
        margin=dict(l=10,r=60,t=10,b=10), height=420,
        xaxis=dict(title="Rentabilidad Media Diaria (%)", gridcolor=COLORES["surface2"]),
        yaxis=dict(gridcolor=COLORES["surface2"]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tabla dinámica ────────────────────────────────────────────────────────
# ── Tabla dinámica ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📋  DIRECTORIO DE FONDOS</div>',
                unsafe_allow_html=True)

    buscar = st.text_input("🔍 Buscar fondo o entidad…", placeholder="Escribe para filtrar…")
    if buscar:
        mask = (
            df_f["NOMBRE_PATRIMONIO"].str.upper().str.contains(buscar.upper(), na=False) |
            df_f["NOMBRE_ENTIDAD"].str.upper().str.contains(buscar.upper(), na=False)
        )
        df_show = df_f[mask]
    else:
        df_show = df_f

    orden   = st.radio("Ordenar por:",
                       ["Rentabilidad EA ↓", "Volatilidad Anual ↑", "Ratio Retorno/Riesgo ↓", "Patrimonio ↓"],
                       horizontal=True)
    sort_map = {
        "Rentabilidad EA ↓"     : ("RENTABILIDAD_ANUAL_EA", False),
        "Volatilidad Anual ↑"   : ("VOLATILIDAD_ANUAL",     True),
        "Ratio Retorno/Riesgo ↓": ("RATIO_RETORNO_RIESGO",  False),
        "Patrimonio ↓"          : ("PATRIMONIO_MEDIO",      False),
    }
    scol, sasc = sort_map[orden]
    df_show = df_show.sort_values(scol, ascending=sasc)

    label_map_p = {"P1": CLUSTER_META["P1"]["label"],
                   "P2": CLUSTER_META["P2"]["label"],
                   "P3": CLUSTER_META["P3"]["label"]}

    tabla = df_show[[
        "NOMBRE_PATRIMONIO", "NOMBRE_ENTIDAD", "PERFIL",
        "RENTABILIDAD_ANUAL_EA", "VOLATILIDAD_ANUAL",
        "RATIO_RETORNO_RIESGO", "PATRIMONIO_MEDIO",
        "INVERSIONISTAS_MEDIO", "DIAS_OBSERVADOS",
    ]].copy()
    
    tabla["PERFIL"]                = tabla["PERFIL"].map(label_map_p)
    tabla["RENTABILIDAD_ANUAL_EA"] = tabla["RENTABILIDAD_ANUAL_EA"].round(2)
    tabla["VOLATILIDAD_ANUAL"]     = tabla["VOLATILIDAD_ANUAL"].round(2)
    tabla["RATIO_RETORNO_RIESGO"]  = tabla["RATIO_RETORNO_RIESGO"].round(4)
    tabla["PATRIMONIO_MEDIO"]      = tabla["PATRIMONIO_MEDIO"].apply(lambda v: fmt_cop(v))
    tabla["INVERSIONISTAS_MEDIO"]  = tabla["INVERSIONISTAS_MEDIO"].apply(lambda v: fmt_num(v))
    
    tabla.columns = [
        "Fondo", "Entidad", "Perfil",
        "Rent. Anual EA %", "Volatilidad Anual %",
        "Ratio Ret/Riesgo", "Patrimonio",
        "Inversionistas", "Días Obs.",
    ]
    st.dataframe(
        tabla.reset_index(drop=True),
        use_container_width = True,
        height              = 500,
    )
    st.caption(f"{len(tabla):,} fondos mostrados")


# ──────────────────────────────────────────────────────────────────────────────
# 7. PESTAÑA 3 — RECOMENDADOR DE PORTAFOLIO
# ──────────────────────────────────────────────────────────────────────────────
def tab_recomendador(grp: pd.DataFrame, output_dir: str | None):

    st.markdown('<div class="section-hdr">🤖  WIZARD DE ASESORÍA — PERFIL DEL CLIENTE</div>',
                unsafe_allow_html=True)

    # ── Formulario de entrada ─────────────────────────────────────────────────
# ── Formulario de entrada ─────────────────────────────────────────────────
    with st.form("wizard_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<span class="step-num">1</span><span class="step-label">Datos del Cliente</span>',
                        unsafe_allow_html=True)
            edad     = st.slider("Edad del inversionista", 18, 80, 35, step=1)
            
            # <- AQUÍ ESTABA EL ERROR: quitamos el "if monto > 0" que rompía la ejecución inicial
            monto    = st.number_input("Monto a invertir ($ COP)",
                                       min_value=0.0, value=5_000_000.0,
                                       step=500_000.0, format="%.0f")
            st.caption(f"💵 Monto configurado para el portafolio")
            
        with c2:
            st.markdown('<span class="step-num">2</span><span class="step-label">Perfil & Horizonte</span>',
                        unsafe_allow_html=True)
            perfil_cli   = st.selectbox("Perfil de riesgo", ["Conservador", "Moderado", "Arriesgado / Crecimiento"])
            horizonte    = st.selectbox("Plazo de inversión",
                                        ["Corto Plazo (<1 año)", "Mediano Plazo (1–3 años)", "Largo Plazo (>3 años)"])

        submitted = st.form_submit_button("✨  Generar Recomendación de Portafolio",
                                          use_container_width=True)

    if not submitted:
        # Preview vacío con instrucciones
        st.info("👆  Completa el formulario y haz clic en **Generar Recomendación** para ver tu portafolio personalizado.")
        return

    # ── Lógica de allocación ──────────────────────────────────────────────────
    # Factor de ajuste por edad (>60 más conservador)
    factor_edad = 0.6 if edad >= 60 else 0.8 if edad >= 45 else 1.0

    alloc_base = {
        ("Conservador",              "Corto Plazo (<1 año)")         : {"P2":0.80,"P3":0.20,"P1":0.00},
        ("Conservador",              "Mediano Plazo (1–3 años)")     : {"P2":0.75,"P3":0.20,"P1":0.05},
        ("Conservador",              "Largo Plazo (>3 años)")        : {"P2":0.70,"P3":0.25,"P1":0.05},
        ("Moderado",                 "Corto Plazo (<1 año)")         : {"P2":0.65,"P3":0.30,"P1":0.05},
        ("Moderado",                 "Mediano Plazo (1–3 años)")     : {"P2":0.60,"P3":0.30,"P1":0.10},
        ("Moderado",                 "Largo Plazo (>3 años)")        : {"P2":0.55,"P3":0.30,"P1":0.15},
        ("Arriesgado / Crecimiento", "Corto Plazo (<1 año)")         : {"P2":0.50,"P3":0.35,"P1":0.15},
        ("Arriesgado / Crecimiento", "Mediano Plazo (1–3 años)")     : {"P2":0.45,"P3":0.30,"P1":0.25},
        ("Arriesgado / Crecimiento", "Largo Plazo (>3 años)")        : {"P2":0.40,"P3":0.25,"P1":0.35},
    }
    alloc = alloc_base.get((perfil_cli, horizonte), {"P2":0.60,"P3":0.30,"P1":0.10})

    # Ajuste por edad: reducir exposición a P1 y redistribuir a P2
    extra_p1   = alloc["P1"] * (1 - factor_edad)
    alloc["P1"] = round(alloc["P1"] * factor_edad, 4)
    alloc["P2"] = round(alloc["P2"] + extra_p1, 4)
    # Normalizar a 100%
    total_alloc = sum(alloc.values())
    alloc = {k: round(v / total_alloc, 4) for k, v in alloc.items()}

    montos = {k: monto * v for k, v in alloc.items()}

    # ── Banner de perfil ──────────────────────────────────────────────────────
    perfil_cfg = {
        "Conservador"              : ("🟢", COLORES["accent"],  "Protección del capital. Baja tolerancia al riesgo."),
        "Moderado"                 : ("🟡", COLORES["warning"], "Equilibrio rentabilidad/riesgo. Horizonte mixto."),
        "Arriesgado / Crecimiento" : ("🔴", COLORES["danger"],  "Maximizar crecimiento. Alta tolerancia al riesgo."),
    }
    emoji_p, color_p, desc_p = perfil_cfg[perfil_cli]

    st.markdown(f"""
    <div style="background:{COLORES['surface2']};border-radius:14px;
                padding:20px 24px;border-left:5px solid {color_p};margin-bottom:20px">
      <span style="font-size:28px">{emoji_p}</span>
      <span style="font-size:18px;font-weight:800;color:{COLORES['text']};margin-left:10px">
        {perfil_cli} · {horizonte}
      </span><br>
      <span style="font-size:13px;color:{COLORES['muted']};margin-left:42px">
        {desc_p} &nbsp;|&nbsp; Edad: {edad} años &nbsp;|&nbsp;
        Monto: {fmt_cop(monto)}
      </span>
    </div>""", unsafe_allow_html=True)

    # ── Asset Allocation (dona) ───────────────────────────────────────────────
    col_dona, col_detalle = st.columns([1, 2])

    with col_dona:
        st.markdown('<div class="section-hdr">💼  ASSET ALLOCATION</div>',
                    unsafe_allow_html=True)
        labels_pie = [CLUSTER_META[k]["label"] for k in ["P2","P3","P1"]]
        values_pie = [alloc["P2"], alloc["P3"], alloc["P1"]]
        colors_pie = [COLORES["perfil2"], COLORES["perfil3"], COLORES["perfil1"]]

        fig_dona = go.Figure(go.Pie(
            labels       = labels_pie,
            values       = values_pie,
            hole         = 0.60,
            marker       = dict(colors=colors_pie, line=dict(color=COLORES["bg"], width=3)),
            texttemplate = "%{label}<br><b>%{percent}</b>",
            textposition = "outside",
            textfont     = dict(size=11),
        ))
        fig_dona.add_annotation(
            text       = f"<b>{fmt_cop(monto)}</b><br><span style='font-size:10px'>Inversión Total</span>",
            x=0.5, y=0.5, showarrow=False,
            font       = dict(size=13, color=COLORES["text"]),
            align      = "center",
        )
        fig_dona.update_layout(
            plot_bgcolor=COLORES["surface"], paper_bgcolor=COLORES["surface"],
            font_color=COLORES["text"], showlegend=False,
            margin=dict(l=10,r=10,t=10,b=10), height=300,
        )
        st.plotly_chart(fig_dona, use_container_width=True)

        # Tabla de montos
        for pkey in ["P2","P3","P1"]:
            meta_p = CLUSTER_META[pkey]
            pct    = alloc[pkey] * 100
            monto_p = montos[pkey]
            st.markdown(f"""
            <div style="background:{COLORES['surface2']};border-radius:8px;
                        padding:9px 14px;margin-bottom:6px;
                        border-left:3px solid {meta_p['color']}">
              <span style="color:{meta_p['color']};font-weight:700;font-size:12px">
                {meta_p['emoji']} {pct:.0f}%
              </span>
              <span style="color:{COLORES['text']};font-size:12px;float:right">
                <b>{fmt_cop(monto_p)}</b>
              </span><br>
              <span style="color:{COLORES['muted']};font-size:11px">{meta_p['label']}</span>
            </div>""", unsafe_allow_html=True)

    with col_detalle:
        st.markdown('<div class="section-hdr">🏆  FONDOS SUGERIDOS POR PERFIL</div>',
                    unsafe_allow_html=True)

        # Selección de fondos: Top N por Ratio Retorno/Riesgo dentro de cada perfil
        fondos_recomendados = []
        n_por_perfil = {"P2": 4, "P3": 2, "P1": 2}

        for pkey in ["P2", "P3", "P1"]:
            if alloc[pkey] == 0:
                continue
            sub = grp[grp["PERFIL"] == pkey].copy()
            if pkey == "P1":
                # Para Perfil 1 (riesgo) recomendar los de menor volatilidad dentro del perfil
                top = sub.nsmallest(n_por_perfil[pkey], "VOLATILIDAD_DIARIA")
            else:
                top = sub.nlargest(n_por_perfil[pkey], "RATIO_RETORNO_RIESGO")

            for _, row in top.iterrows():
                fondos_recomendados.append({
                    "perfil"    : pkey,
                    "fondo"     : row["NOMBRE_PATRIMONIO"],
                    "entidad"   : row["NOMBRE_ENTIDAD"],
                    "rent"      : row["RENTABILIDAD_MEDIA"],
                    "vol"       : row["VOLATILIDAD_DIARIA"],
                    "ratio"     : row["RATIO_RETORNO_RIESGO"],
                    "patrimonio": row["PATRIMONIO_MEDIO"],
                    "pct_port"  : alloc[pkey] / n_por_perfil[pkey],
                })

        for rec in fondos_recomendados:
            meta_p   = CLUSTER_META[rec["perfil"]]
            monto_rec = monto * rec["pct_port"]
            color_rent = COLORES["accent"] if rec["rent"] >= 0 else COLORES["danger"]
            st.markdown(f"""
            <div class="rec-card" style="border-top-color:{meta_p['color']}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div class="rec-title">{rec['fondo'][:60]}</div>
                  <div class="rec-ent">{meta_p['emoji']} {meta_p['label']} &nbsp;·&nbsp; {rec['entidad'][:35]}</div>
                </div>
                <div style="text-align:right;flex-shrink:0;margin-left:12px">
                  <div style="font-size:14px;font-weight:800;color:{color_rent}">
                    {fmt_pct(rec['rent'], sign=True)}
                  </div>
                  <div style="font-size:10px;color:{COLORES['muted']}">Rent. Diaria</div>
                </div>
              </div>
              <div style="display:flex;gap:20px;margin-top:10px;flex-wrap:wrap">
                <span class="rec-metric">Monto sugerido: {fmt_cop(monto_rec)}</span>
                <span style="color:{COLORES['muted']};font-size:12px">Vol: {fmt_pct(rec['vol'])}</span>
                <span style="color:{COLORES['muted']};font-size:12px">Ratio: {rec['ratio']:.4f}</span>
                <span style="color:{COLORES['muted']};font-size:12px">AUM: {fmt_cop(rec['patrimonio'])}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Proyección de capital ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📈  PROYECCIÓN ESTIMADA DE CAPITAL</div>',
                unsafe_allow_html=True)
    st.caption("⚠️  Proyección ilustrativa basada en rentabilidad anual E.A. histórica del portafolio. Las rentabilidades pasadas no garantizan resultados futuros.")

    # ── Rentabilidad anual ponderada del portafolio ───────────────────────────
    # RENTABILIDAD_MEDIA está en % anual E.A. (ej: 9.98 significa 9.98% E.A.)
    # NO aplicar compuesto diario — es una tasa ya anualizada.
    rent_port_pct = sum(                          # en %, ej: 9.98
        alloc[pk] * grp[grp["PERFIL"] == pk]["RENTABILIDAD_MEDIA"].mean()
        for pk in ["P1","P2","P3"] if not grp[grp["PERFIL"]==pk].empty
    )

    # Volatilidad anual del portafolio (también en % para el rango de escenarios)
    vol_port_pct  = sum(
        alloc[pk] * grp[grp["PERFIL"]==pk]["VOLATILIDAD_DIARIA"].mean()
        for pk in ["P1","P2","P3"] if not grp[grp["PERFIL"]==pk].empty
    )

    # Horizonte en años (con pasos mensuales para la curva)
    anos_map  = {"Corto Plazo (<1 año)": 1, "Mediano Plazo (1–3 años)": 3, "Largo Plazo (>3 años)": 5}
    anos_proy = anos_map.get(horizonte, 1)

    # Eje X en meses para que la curva sea legible
    n_pasos  = anos_proy * 12          # un punto por mes
    t_meses  = np.arange(0, n_pasos + 1)
    t_anos   = t_meses / 12.0          # fracción de año para la potencia

    # Fórmula correcta: monto * (1 + tasa_decimal) ** t_años
    # donde tasa_decimal = rent_port_pct / 100
    val_esperado = monto * (1 + rent_port_pct / 100) ** t_anos
    val_opt      = monto * (1 + (rent_port_pct + vol_port_pct) / 100) ** t_anos
    val_cons     = monto * (1 + max(rent_port_pct - vol_port_pct, -50) / 100) ** t_anos

    # Etiquetas del eje X en formato "Mes 1 / Año 1"
    x_labels = [f"M{m}" if m % 12 != 0 else f"Año {m//12}" for m in t_meses]

    fig_proy = go.Figure()
    fig_proy.add_trace(go.Scatter(
        x=t_meses, y=val_opt,
        name="Escenario Optimista",
        line=dict(color=COLORES["accent"], dash="dot", width=1.5),
        fill=None,
    ))
    fig_proy.add_trace(go.Scatter(
        x=t_meses, y=val_cons,
        name="Escenario Conservador",
        line=dict(color=COLORES["danger"], dash="dot", width=1.5),
        fill="tonexty",
        fillcolor="rgba(79,142,247,.08)",
    ))
    fig_proy.add_trace(go.Scatter(
        x=t_meses, y=val_esperado,
        name="Escenario Esperado",
        line=dict(color=COLORES["primary"], width=2.5),
        hovertemplate="<b>%{text}</b><br>Valor: $%{y:,.0f} COP<extra></extra>",
        text=x_labels,
    ))
    fig_proy.add_hline(
        y=monto, line_dash="dash",
        line_color="rgba(255,255,255,.3)",
        annotation_text="Capital Inicial",
        annotation_position="right",
    )
    # Marcas en el eje X cada 12 meses (años completos)
    tickvals = list(range(0, n_pasos + 1, 12))
    ticktext = [f"Año {v//12}" if v > 0 else "Hoy" for v in tickvals]
    fig_proy.update_layout(
        plot_bgcolor=COLORES["surface"], paper_bgcolor=COLORES["surface"],
        font_color=COLORES["text"],
        xaxis=dict(
            title="Horizonte temporal",
            tickvals=tickvals, ticktext=ticktext,
            gridcolor=COLORES["surface2"],
        ),
        yaxis=dict(
            title="Valor del Portafolio (COP)",
            tickformat="$,.0f",
            gridcolor=COLORES["surface2"],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
    )
    st.plotly_chart(fig_proy, use_container_width=True)

    # ── KPIs de resultado ─────────────────────────────────────────────────────
    val_final  = val_esperado[-1]
    val_opt_f  = val_opt[-1]
    val_cons_f = val_cons[-1]
    ganancia   = val_final - monto
    sign_gan   = "+" if ganancia >= 0 else ""

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown(kpi_card(
            "Rentab. Anual Portafolio",
            f"{rent_port_pct:+.2f}% E.A.",
            f"Tasa efectiva anual ponderada por perfil",
            COLORES["accent"] if rent_port_pct >= 0 else COLORES["danger"],
        ), unsafe_allow_html=True)
    with col_r2:
        st.markdown(kpi_card(
            f"Valor Esperado ({anos_proy} año{'s' if anos_proy > 1 else ''})",
            fmt_cop(val_final),
            f"Ganancia: {sign_gan}{fmt_cop(ganancia)}",
            COLORES["primary"],
        ), unsafe_allow_html=True)
    with col_r3:
        st.markdown(kpi_card(
            "Escenario Optimista",
            fmt_cop(val_opt_f),
            f"+{fmt_cop(val_opt_f - monto)} vs capital inicial",
            COLORES["accent"],
        ), unsafe_allow_html=True)
    with col_r4:
        st.markdown(kpi_card(
            "Escenario Conservador",
            fmt_cop(val_cons_f),
            f"{fmt_cop(val_cons_f - monto)} vs capital inicial",
            COLORES["danger"] if val_cons_f < monto else COLORES["accent"],
        ), unsafe_allow_html=True)


    # ── Exportar resumen ejecutivo ────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">📄  EXPORTAR RESUMEN EJECUTIVO</div>',
                unsafe_allow_html=True)

    resumen_txt = generar_resumen_texto(
        perfil_cli, horizonte, edad, monto, alloc, fondos_recomendados,
        val_esperado[-1], val_opt[-1], val_cons[-1], rent_port_pct,
    )

    col_dl1, col_dl2, _ = st.columns([1, 1, 2])
    with col_dl1:
        st.download_button(
            label     = "⬇️  Descargar Resumen (.txt)",
            data      = resumen_txt,
            file_name = f"Recomendacion_FIC_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime      = "text/plain",
            use_container_width = True,
        )

    # CSV de los fondos recomendados
    df_rec_exp = pd.DataFrame(fondos_recomendados)
    df_rec_exp["monto_sugerido"] = df_rec_exp["pct_port"] * monto
    csv_rec = df_rec_exp.to_csv(index=False).encode("utf-8-sig")
    with col_dl2:
        st.download_button(
            label     = "⬇️  Fondos Recomendados (.csv)",
            data      = csv_rec,
            file_name = f"Fondos_Recomendados_FIC_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime      = "text/csv",
            use_container_width = True,
        )

    if output_dir:
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            path_txt = os.path.join(output_dir, f"Recomendacion_FIC_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
            with open(path_txt, "w", encoding="utf-8") as f:
                f.write(resumen_txt)
            st.success(f"✅ Resumen guardado en: `{path_txt}`")
        except Exception as e:
            st.warning(f"No se pudo guardar en la ruta configurada: {e}")

    # Disclaimer
    st.markdown(f"""
    <div style="background:{COLORES['surface2']};border:1px solid {COLORES['warning']};
                border-radius:10px;padding:14px 18px;margin-top:16px;font-size:12px;
                color:{COLORES['muted']}">
      ⚠️ <strong>Aviso legal:</strong> Este recomendador es una herramienta de análisis educativa
      basada en datos históricos. Las proyecciones son ilustrativas y no constituyen asesoría
      financiera certificada. Rentabilidades pasadas no garantizan resultados futuros.
      Consulta un asesor financiero certificado antes de invertir.
    </div>""", unsafe_allow_html=True)


def generar_resumen_texto(perfil, horizonte, edad, monto, alloc,
                           fondos, val_esp, val_opt, val_cons, rent_port) -> str:
    sep = "=" * 62
    linea = "-" * 62
    lines = [
        sep,
        "  RESUMEN EJECUTIVO — RECOMENDACIÓN DE PORTAFOLIO FIC",
        f"  Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        sep, "",
        "PERFIL DEL CLIENTE",
        linea,
        f"  Perfil de Riesgo     : {perfil}",
        f"  Horizonte de Inversión: {horizonte}",
        f"  Edad                 : {edad} años",
        f"  Monto a Invertir     : {fmt_cop(monto)} COP",
        "",
        "ASSET ALLOCATION PROPUESTO",
        linea,
        f"  🟢 Fondos Sólidos (P2)          : {alloc['P2']*100:.0f}%  —  {fmt_cop(monto*alloc['P2'])}",
        f"  🔵 Mega Fondos Institucionales (P3): {alloc['P3']*100:.0f}%  —  {fmt_cop(monto*alloc['P3'])}",
        f"  🔴 Riesgo Elevado (P1)           : {alloc['P1']*100:.0f}%  —  {fmt_cop(monto*alloc['P1'])}",
        "",
        "FONDOS SUGERIDOS",
        linea,
    ]
    for r in fondos:
        lines.append(f"  [{CLUSTER_META[r['perfil']]['emoji']} {r['perfil']}] {r['fondo'][:52]}")
        lines.append(f"    Entidad: {r['entidad'][:40]}")
        lines.append(f"    Monto: {fmt_cop(monto * r['pct_port'])}  |  Rent.Diaria: {fmt_pct(r['rent'])}  |  Vol: {fmt_pct(r['vol'])}")
        lines.append("")
    lines += [
        "PROYECCIÓN DE CAPITAL",
        linea,
        f"  Rentabilidad anual E.A. del portafolio: {fmt_pct(rent_port)}",
        f"  Valor Esperado Final               : {fmt_cop(val_esp)}",
        f"  Escenario Optimista                : {fmt_cop(val_opt)}",
        f"  Escenario Conservador              : {fmt_cop(val_cons)}",
        "",
        sep,
        "  AVISO: Documento informativo. No constituye asesoría financiera.",
        "  Rentabilidades pasadas no garantizan resultados futuros.",
        sep,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 8. SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar(meta: dict | None, output_dir_state):
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:16px 0 8px;text-align:center">
          <div style="font-size:32px">📊</div>
          <div style="font-size:16px;font-weight:800;color:{COLORES['text']}">FIC Advisor 2026</div>
          <div style="font-size:11px;color:{COLORES['muted']}">Dashboard · Superfinanciera Colombia</div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        if meta:
            st.markdown(f"<div class='section-hdr'>DATOS CARGADOS</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:12px;color:{COLORES['muted']}">
              📁 <b style="color:{COLORES['text']}">{meta['filename'][:28]}…</b><br><br>
              📦 <b>{meta['n_fondos']:,}</b> fondos analizados<br>
              🏦 <b>{meta['n_entidades']}</b> entidades<br>
              📅 {meta['fecha_min'].strftime('%d %b %Y') if meta['fecha_min'] else '—'} →
                 {meta['fecha_max'].strftime('%d %b %Y') if meta['fecha_max'] else '—'}
            </div>""", unsafe_allow_html=True)
            st.divider()

        st.markdown(f"<div class='section-hdr'>CONFIGURACIÓN</div>", unsafe_allow_html=True)
        new_dir = st.text_input(
            "📂 Directorio de exportación (opcional)",
            value       = output_dir_state or "",
            placeholder = "/ruta/donde/guardar",
            help        = "Ruta local donde se guardarán los reportes descargados.",
        )

        st.divider()
        st.markdown(f"""
        <div style="font-size:11px;color:{COLORES['muted']};line-height:1.7">
          <b style="color:{COLORES['text']}">Metodología</b><br>
          Limpieza → Filtro ABIERTOS<br>
          Agregación por ID_FONDO_SERIE<br>
          K-Means k=3 (sin data leakage)<br>
          Rf = 0 (proxy Sharpe diario)<br><br>
          <b style="color:{COLORES['text']}">Colores</b><br>
          <span style="color:{COLORES['perfil2']}">● Sólidos / Estables</span><br>
          <span style="color:{COLORES['perfil3']}">● Mega Fondos</span><br>
          <span style="color:{COLORES['perfil1']}">● Riesgo Elevado</span>
        </div>""", unsafe_allow_html=True)

    return new_dir or None


# ──────────────────────────────────────────────────────────────────────────────
# 9. ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Limpiar caché al primer arranque de la sesión para evitar
    # que resultados de ejecuciones anteriores contaminen la vista.
    if "cache_cleared" not in st.session_state:
        st.cache_data.clear()
        st.session_state["cache_cleared"] = True

    # Estado de sesión
    if "output_dir" not in st.session_state:
        st.session_state["output_dir"] = None

    output_dir = render_sidebar(
        st.session_state.get("meta"),
        st.session_state["output_dir"],
    )
    if output_dir != st.session_state["output_dir"]:
        st.session_state["output_dir"] = output_dir

    # Pantalla de bienvenida / carga
    if "grp" not in st.session_state:
        archivo = pantalla_bienvenida()
        if archivo is not None:
            with st.spinner("⚙️  Ejecutando pipeline de datos (limpieza → clustering)…"):
                try:
                    df_det, grp, meta = cargar_y_procesar(archivo.read(), archivo.name)
                    st.session_state["df_det"] = df_det
                    st.session_state["grp"]    = grp
                    st.session_state["meta"]   = meta
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo: **{e}**")
                    st.markdown("""
                    **Posibles causas:**
                    - El archivo no es el CSV oficial de la Superfinanciera
                    - El archivo tiene columnas faltantes o formato incorrecto
                    - Encoding no soportado

                    Verifica que el archivo sea `Rentabilidades_de_los_Fondos_de_Inversión_Colectiva_(FIC)_*.csv`
                    """)
        return

    # Dashboard principal
    grp  = st.session_state["grp"]
    meta = st.session_state["meta"]

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px">
      <div style="font-size:28px">📊</div>
      <div>
        <div style="font-size:20px;font-weight:800;color:{COLORES['text']}">
          Dashboard FIC · Asesoría de Inversión
        </div>
        <div style="font-size:12px;color:{COLORES['muted']}">
          {meta['n_fondos']:,} fondos analizados · {meta['n_entidades']} entidades ·
          Corte: {meta['fecha_max'].strftime('%d %b %Y') if meta['fecha_max'] else '—'}
        </div>
      </div>
      <div style="margin-left:auto">
    """, unsafe_allow_html=True)

    if st.button("🔄  Cargar otro archivo", key="reset_btn"):
        for k in ["df_det","grp","meta"]:
            st.session_state.pop(k, None)
        st.cache_data.clear()
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Tabs principales
    t1, t2, t3 = st.tabs([
        "📌 Visión General del Mercado",
        "🗂️  Explorador de Clusters",
        "🤖 Recomendador de Portafolio",
    ])

    with t1:
        tab_vision_general(st.session_state["df_det"], grp, meta)
    with t2:
        tab_explorador(grp)
    with t3:
        tab_recomendador(grp, st.session_state["output_dir"])


if __name__ == "__main__":
    main()
