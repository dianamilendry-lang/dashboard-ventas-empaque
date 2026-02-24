# app.py
# Dashboard Mensual + IA Generativa (Gemini) + Asistente Técnico Preventa (RAG simple sin vector DB)
#
# ✅ NO usa reportlab (evita ModuleNotFoundError)
# ✅ Soporta Excel mensual (ventas/budget) aunque solo vengan 1–2 meses con datos
# ✅ IA del Dashboard (conclusiones + recomendaciones) usando Gemini
# ✅ Forecast & Riesgo (sin librerías extra)
# ✅ Asistente técnico basado en PDF del manual (RAG simple por “chunks” + scoring)
#
# Requisitos (requirements.txt):
# streamlit
# pandas
# openpyxl
# plotly
# PyPDF2
# google-genai

import os
import re
import math
import textwrap
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader

# Google Gen AI SDK
from google import genai
from google.genai.types import HttpOptions

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard Mensual + IA Generativa", layout="wide")

APP_TITLE = "📊 Dashboard Mensual + IA Generativa"

MESES_ORDEN = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

MESES_VENTAS = {
    "Enero": "Ene_KG",
    "Febrero": "Feb_KG",
    "Marzo": "Mar_KG",
    "Abril": "Abr_KG",
    "Mayo": "May_KG",
    "Junio": "Jun_KG",
    "Julio": "Jul_KG",
    "Agosto": "Ago_KG",
    "Septiembre": "Sep_KG",
    "Octubre": "Oct_KG",
    "Noviembre": "Nov_KG",
    "Diciembre": "Dic_KG",
}

MESES_PRES = {
    "Enero": "ENE", "Febrero": "FEB", "Marzo": "MAR", "Abril": "ABR",
    "Mayo": "MAY", "Junio": "JUN", "Julio": "JUL", "Agosto": "AGO",
    "Septiembre": "SEP", "Octubre": "OCT", "Noviembre": "NOV", "Diciembre": "DIC",
}

# Carpeta recomendada en la repo:
# manual_tecnico/Manual_tecnico_preventa.pdf
MANUAL_DIR = "manual_tecnico"
MANUAL_FALLBACK_ROOT_NAMES = [
    "Manual_tecnico_preventa.pdf",
    "manual_tecnico_preventa.pdf",
    "Manual tecnico preventa.pdf",
]

# Modelos: por cambios frecuentes, intentamos una lista hasta que funcione.
DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.5-pro",
]

# =========================
# UTIL: GEMINI
# =========================
def _get_google_api_key() -> str | None:
    # Acepta varias llaves para que no se te rompa por naming
    # Recomendado por Google: GOOGLE_API_KEY
    return (
        st.secrets.get("GOOGLE_API_KEY")
        or st.secrets.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )

@st.cache_resource(show_spinner=False)
def gemini_client():
    api_key = _get_google_api_key()
    if not api_key:
        return None
    # api_version="v1" ayuda a evitar algunos errores v1beta en entornos.
    return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))

def gemini_generate(prompt: str, model_preference: str | None = None) -> tuple[str | None, str | None]:
    """
    Devuelve (texto, error). Si falla, texto=None y error con detalle.
    """
    client = gemini_client()
    if client is None:
        return None, "Falta GOOGLE_API_KEY (o GEMINI_API_KEY) en Streamlit Secrets."

    candidates = []
    if model_preference:
        candidates.append(model_preference.strip())
    candidates.extend([m for m in DEFAULT_MODEL_CANDIDATES if m not in candidates])

    last_err = None
    for model in candidates:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            txt = getattr(resp, "text", None)
            if txt:
                return txt, None
            # fallback si viene en otro formato
            return str(resp), None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    return None, f"No pude generar respuesta con Gemini. Último error: {last_err}"


# =========================
# UTIL: MANUAL PDF (RAG simple)
# =========================
@st.cache_data(show_spinner=False)
def find_manual_pdf() -> str | None:
    # 1) Busca en manual_tecnico/
    if os.path.isdir(MANUAL_DIR):
        pdfs = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
        pdfs.sort()
        if pdfs:
            return os.path.join(MANUAL_DIR, pdfs[0])

    # 2) Busca en raíz con nombres comunes
    root_files = set(os.listdir("."))
    for name in MANUAL_FALLBACK_ROOT_NAMES:
        if name in root_files:
            return name

    # 3) Busca cualquier PDF en raíz
    pdfs_root = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    pdfs_root.sort()
    if pdfs_root:
        return pdfs_root[0]

    return None

@st.cache_data(show_spinner=False)
def load_manual_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j]
        chunks.append(chunk.strip())
        i = max(j - overlap, i + 1)
    return [c for c in chunks if c]

def _tokenize(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñ0-9\s\-\/]", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    return toks

def retrieve_chunks(query: str, chunks: list[str], k: int = 5) -> list[tuple[float, str]]:
    """
    Scoring simple por overlap + bonus por frases.
    """
    q_tokens = _tokenize(query)
    if not q_tokens or not chunks:
        return []

    q_set = set(q_tokens)
    scored = []
    for c in chunks:
        c_tokens = _tokenize(c)
        if not c_tokens:
            continue
        c_set = set(c_tokens)
        overlap = len(q_set & c_set)
        denom = math.sqrt(len(q_set) * len(c_set)) if (len(q_set) and len(c_set)) else 1.0
        score = overlap / denom

        # Bonus por coincidencias “clave”
        bonus = 0.0
        for phrase in ["vffs", "hffs", "zipper", "válvula", "valvula", "aluminio", "metpet", "pet", "pe", "bop", "sellado"]:
            if phrase in query.lower() and phrase in c.lower():
                bonus += 0.08
        scored.append((score + bonus, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

def short_context(query: str, manual_chunks: list[str], k: int = 5, max_chars: int = 6000) -> str:
    top = retrieve_chunks(query, manual_chunks, k=k)
    ctx = []
    total = 0
    for score, chunk in top:
        piece = f"[EVIDENCIA score={score:.3f}]\n{chunk}\n"
        if total + len(piece) > max_chars:
            break
        ctx.append(piece)
        total += len(piece)
    return "\n\n".join(ctx).strip()


# =========================
# UTIL: DATA NORMALIZATION
# =========================
def detect_months_with_data(df_ventas: pd.DataFrame) -> list[str]:
    """
    Detecta meses que realmente traen datos (no solo columnas vacías/0).
    Si tu Excel mensual solo tiene 1–2 meses con números, el dashboard NO castigará el resto.
    """
    available = []
    for mes, col in MESES_VENTAS.items():
        if col not in df_ventas.columns:
            continue
        ser = pd.to_numeric(df_ventas[col], errors="coerce")
        # “Hay datos” si existe al menos un valor no nulo y la suma absoluta > 0
        if ser.notna().any() and float(ser.fillna(0).abs().sum()) > 0:
            available.append(mes)
    return available

def normalizar_ventas(df: pd.DataFrame, meses_activos: list[str]) -> pd.DataFrame:
    base_cols = []
    for c in ["SlpName", "Código de cliente/proveedor", "Nombre de cliente", "ItemCode", "ItemName", "UM"]:
        if c in df.columns:
            base_cols.append(c)

    if "ItemCode" not in df.columns:
        raise ValueError("Tu reporte de ventas debe tener la columna 'ItemCode'.")

    out = []
    for mes in meses_activos:
        col = MESES_VENTAS[mes]
        if col not in df.columns:
            continue
        tmp = df[base_cols].copy() if base_cols else df[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["actual_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out.append(tmp)

    if not out:
        # si no detectamos meses activos, intenta con TODOS los meses presentes en columnas (aunque sean 0)
        for mes, col in MESES_VENTAS.items():
            if col in df.columns:
                tmp = df[base_cols].copy() if base_cols else df[["ItemCode"]].copy()
                tmp["mes"] = mes
                tmp["actual_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                out.append(tmp)

    return pd.concat(out, ignore_index=True)

def normalizar_presupuesto(df: pd.DataFrame, meses_activos: list[str]) -> pd.DataFrame:
    if "ItemCode" not in df.columns:
        raise ValueError("Tu presupuesto debe tener la columna 'ItemCode'.")

    out = []
    for mes in meses_activos:
        col = MESES_PRES.get(mes)
        if not col:
            continue
        if col not in df.columns:
            # si presupuesto no trae esa columna, asumimos 0 para ese mes
            tmp = df[["ItemCode"]].copy()
            tmp["mes"] = mes
            tmp["budget_kg"] = 0.0
            out.append(tmp)
            continue

        tmp = df[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["budget_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out.append(tmp)

    return pd.concat(out, ignore_index=True)

def build_df_final(dfv: pd.DataFrame, dfp: pd.DataFrame, anio: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    meses_activos = detect_months_with_data(dfv)

    # Si el reporte viene “mensual” y solo incluye 1 mes, esto lo soporta.
    # Si no detecta meses, igual intenta normalizar por columnas existentes.
    ventas_long = normalizar_ventas(dfv, meses_activos if meses_activos else MESES_ORDEN)
    pres_long = normalizar_presupuesto(dfp, meses_activos if meses_activos else MESES_ORDEN)

    df = ventas_long.merge(pres_long, on=["ItemCode", "mes"], how="left")
    df["budget_kg"] = df["budget_kg"].fillna(0)
    df["var_kg"] = df["actual_kg"] - df["budget_kg"]
    df["cumpl_pct"] = df.apply(lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0.0, axis=1)

    df["mes"] = pd.Categorical(df["mes"], categories=MESES_ORDEN, ordered=True)

    if anio is None:
        anio = datetime.now().year
    df["anio"] = int(anio)

    # meses “analizables”: los que detectamos con datos, si no, los que aparezcan
    meses_analizables = meses_activos if meses_activos else sorted(df["mes"].dropna().astype(str).unique(), key=lambda m: MESES_ORDEN.index(m))
    return df, meses_analizables


# =========================
# FORECAST & RISK (sin deps extra)
# =========================
def simple_forecast(series_by_month: pd.Series, steps: int = 3) -> pd.DataFrame:
    """
    Forecast naive: promedio móvil de los últimos 2 meses con datos (>0).
    Si solo hay 1 mes, replica ese valor.
    """
    s = series_by_month.copy().astype(float)
    s = s.reindex(MESES_ORDEN)
    nonzero = [v for v in s.values if (v is not None and not pd.isna(v) and v > 0)]
    if len(nonzero) >= 2:
        base = sum(nonzero[-2:]) / 2
    elif len(nonzero) == 1:
        base = nonzero[-1]
    else:
        base = 0.0

    # próximo mes después del último mes con data
    last_idx = None
    for i, m in enumerate(MESES_ORDEN):
        v = s.loc[m]
        if (v is not None) and (not pd.isna(v)) and (v > 0):
            last_idx = i
    if last_idx is None:
        last_idx = -1

    rows = []
    for k in range(1, steps + 1):
        idx = last_idx + k
        if idx >= len(MESES_ORDEN):
            mes = f"Mes+{k}"
        else:
            mes = MESES_ORDEN[idx]
        rows.append({"mes": mes, "forecast_actual_kg": base})
    return pd.DataFrame(rows)

def risk_summary(df: pd.DataFrame, meses_usados: list[str]) -> dict:
    """
    Genera riesgos ejecutivos simples:
    - Calidad de datos (meses sin ventas)
    - Underperformance vs budget
    - Concentración por vendedor/cliente (si existen columnas)
    """
    dff = df.copy()
    dff["mes_str"] = dff["mes"].astype(str)
    dff = dff[dff["mes_str"].isin(meses_usados)]

    total_actual = float(dff["actual_kg"].sum())
    total_budget = float(dff["budget_kg"].sum())
    cumplimiento = (total_actual / total_budget * 100) if total_budget > 0 else 0.0
    under = float((dff["var_kg"] < 0).sum()) / max(len(dff), 1)

    # Meses con 0 actual dentro de los meses usados
    by_mes = dff.groupby("mes_str", as_index=False)[["actual_kg", "budget_kg"]].sum()
    meses_cero = by_mes[by_mes["actual_kg"] <= 0]["mes_str"].tolist()

    # Concentración por vendedor
    conc_vend = None
    if "SlpName" in dff.columns:
        by_v = dff.groupby("SlpName", as_index=False)["actual_kg"].sum().sort_values("actual_kg", ascending=False)
        if not by_v.empty and total_actual > 0:
            top_share = float(by_v.iloc[0]["actual_kg"]) / total_actual * 100
            conc_vend = {"top_vendedor": str(by_v.iloc[0]["SlpName"]), "top_share_pct": top_share}

    # Concentración por cliente
    conc_cli = None
    cli_col = None
    for c in ["Nombre de cliente", "Código de cliente/proveedor"]:
        if c in dff.columns:
            cli_col = c
            break
    if cli_col:
        by_c = dff.groupby(cli_col, as_index=False)["actual_kg"].sum().sort_values("actual_kg", ascending=False)
        if not by_c.empty and total_actual > 0:
            top_share = float(by_c.iloc[0]["actual_kg"]) / total_actual * 100
            conc_cli = {"top_cliente": str(by_c.iloc[0][cli_col]), "top_share_pct": top_share, "col": cli_col}

    # “Score” simple 0–100
    score = 0
    score += 30 if len(meses_cero) > 0 else 0
    score += 30 if cumplimiento < 90 else (10 if cumplimiento < 100 else 0)
    score += 20 if (conc_vend and conc_vend["top_share_pct"] >= 70) else 0
    score += 20 if (conc_cli and conc_cli["top_share_pct"] >= 70) else 0
    score = min(score, 100)

    return {
        "cumplimiento_pct": cumplimiento,
        "meses_cero": meses_cero,
        "under_ratio": under,
        "conc_vend": conc_vend,
        "conc_cli": conc_cli,
        "risk_score_0_100": score,
    }


# =========================
# UI
# =========================
st.title(APP_TITLE)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cargar Excel",
    "Dashboard",
    "IA del Dashboard",
    "Forecast & Riesgo",
    "IA Técnica Preventa",
])

# -------------------------
# TAB 1: CARGA
# -------------------------
with tab1:
    st.subheader("1) Cargar archivos (mensual)")
    colA, colB = st.columns(2)

    with colA:
        ventas_file = st.file_uploader("Reporte de Ventas mensual (.xlsx)", type=["xlsx"], key="ventas_upl")
    with colB:
        pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"], key="pres_upl")

    c1, c2 = st.columns([1, 2])
    with c1:
        anio = st.number_input("Año", value=datetime.now().year, step=1)

    if st.button("Procesar", type="primary"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos: Ventas y Presupuesto.")
        else:
            try:
                dfv = pd.read_excel(ventas_file)
                dfp = pd.read_excel(pres_file)

                df_final, meses_analizables = build_df_final(dfv, dfp, anio=int(anio))
                st.session_state["df_final"] = df_final
                st.session_state["meses_analizables"] = meses_analizables

                st.success("✅ Datos procesados correctamente.")
                st.info(f"Meses detectados con datos reales: {', '.join(meses_analizables) if meses_analizables else 'N/A'}")

                with st.expander("Ver muestra (primeras filas)"):
                    st.dataframe(df_final.head(30), use_container_width=True)

            except Exception as e:
                st.error(f"Error procesando archivos: {type(e).__name__}: {e}")

    st.caption("Tip: si tu Excel mensual solo tiene 1–2 meses con números, este sistema calcula KPIs solo sobre esos meses (no penaliza meses futuros).")


# -------------------------
# SIDEBAR FILTROS (global)
# -------------------------
def sidebar_filters(df: pd.DataFrame, default_meses: list[str]):
    st.sidebar.header("Filtros")

    # Año
    anios = sorted(df["anio"].dropna().unique().tolist())
    anio_sel = st.sidebar.selectbox("Año", anios, index=len(anios) - 1 if anios else 0)
    dff = df[df["anio"] == anio_sel].copy()

    # Mes (por defecto: meses con data real)
    meses_disp = [m for m in MESES_ORDEN if m in dff["mes"].astype(str).unique().tolist()]
    default = [m for m in default_meses if m in meses_disp] or meses_disp
    mes_sel = st.sidebar.multiselect("Mes (para análisis)", meses_disp, default=default)
    if mes_sel:
        dff = dff[dff["mes"].astype(str).isin(mes_sel)].copy()

    # Vendedor (si existe)
    if "SlpName" in dff.columns:
        vend_disp = sorted([v for v in dff["SlpName"].dropna().unique().tolist()])
        if vend_disp:
            vend_sel = st.sidebar.multiselect("Vendedor", vend_disp, default=vend_disp)
            if vend_sel:
                dff = dff[dff["SlpName"].isin(vend_sel)].copy()

    return dff, anio_sel, mes_sel


# -------------------------
# TAB 2: DASHBOARD
# -------------------------
with tab2:
    st.subheader("2) Dashboard (cumplimiento vs presupuesto)")
    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña **Cargar Excel**.")
    else:
        df0 = st.session_state["df_final"].copy()
        meses_default = st.session_state.get("meses_analizables", [])

        dff, anio_sel, mes_sel = sidebar_filters(df0, meses_default)

        total_actual = float(dff["actual_kg"].sum())
        total_budget = float(dff["budget_kg"].sum())
        total_var = total_actual - total_budget
        total_pct = (total_actual / total_budget * 100) if total_budget > 0 else 0.0

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual (KG)", f"{total_actual:,.0f}")
        c2.metric("Budget (KG)", f"{total_budget:,.0f}")
        c3.metric("Varianza (KG)", f"{total_var:,.0f}")
        c4.metric("% Cumplimiento", f"{total_pct:.1f}%")

        st.markdown("### Tendencia mensual (Actual vs Budget)")
        by_mes = dff.groupby(dff["mes"].astype(str), as_index=False)[["actual_kg", "budget_kg"]].sum()
        by_mes.rename(columns={"mes": "mes"}, inplace=True)
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        st.plotly_chart(
            px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True),
            use_container_width=True
        )

        # Cumplimiento por vendedor (si existe)
        if "SlpName" in dff.columns and dff["SlpName"].notna().any():
            st.markdown("### Cumplimiento por vendedor (%)")
            by_vend = dff.groupby("SlpName", as_index=False)[["actual_kg", "budget_kg"]].sum()
            by_vend["cumpl_pct"] = by_vend.apply(
                lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0.0,
                axis=1
            )
            by_vend = by_vend.sort_values("cumpl_pct", ascending=False)
            st.plotly_chart(px.bar(by_vend, x="SlpName", y="cumpl_pct"), use_container_width=True)

        # Export de datos filtrados (sin librerías extra)
        st.markdown("### Exportación")
        csv_data = dff.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar datos filtrados (CSV)",
            data=csv_data,
            file_name=f"dashboard_filtrado_{anio_sel}.csv",
            mime="text/csv"
        )


# -------------------------
# TAB 3: IA DEL DASHBOARD
# -------------------------
with tab3:
    st.subheader("3) IA del Dashboard (análisis ejecutivo)")
    st.caption("Genera un informe SIN inventar cifras, usando solo agregados por mes del filtro actual.")

    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña **Cargar Excel**.")
    else:
        df0 = st.session_state["df_final"].copy()
        meses_default = st.session_state.get("meses_analizables", [])
        dff, anio_sel, mes_sel = sidebar_filters(df0, meses_default)

        model_choice = st.selectbox("Modelo (si falla, prueba otro)", DEFAULT_MODEL_CANDIDATES, index=0)

        by_mes = dff.groupby(dff["mes"].astype(str), as_index=False)[["actual_kg", "budget_kg", "var_kg"]].sum()
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        # tabla compacta para prompt
        resumen_lines = []
        for _, r in by_mes.iterrows():
            m = str(r["mes"])
            a = float(r["actual_kg"])
            b = float(r["budget_kg"])
            v = float(r["var_kg"])
            pct = (a / b * 100) if b > 0 else 0.0
            resumen_lines.append(f"- {m}: Actual={a:,.2f} kg | Budget={b:,.2f} kg | Var={v:,.2f} kg | Cumpl={pct:.1f}%")
        resumen = "\n".join(resumen_lines)

        total_actual = float(dff["actual_kg"].sum())
        total_budget = float(dff["budget_kg"].sum())
        total_pct = (total_actual / total_budget * 100) if total_budget > 0 else 0.0

        if st.button("Generar análisis con IA", type="primary"):
            prompt = f"""
Eres consultor ejecutivo (MBA) en estrategia comercial e inteligencia de negocios para empaques flexibles.

REGLAS:
- Usa SOLO los datos provistos.
- NO inventes cifras.
- Si notas meses sin datos (0), interpreta como "aún no cargados" si el reporte es mensual.

CONTEXTO:
Año: {anio_sel}
Meses analizados (filtro): {", ".join(mes_sel) if mes_sel else "N/A"}
Total Actual: {total_actual:,.2f} kg
Total Budget: {total_budget:,.2f} kg
Cumplimiento total del periodo: {total_pct:.1f}%

DATOS POR MES:
{resumen}

ENTREGA (en español, con bullets):
1) Resumen ejecutivo (5 bullets)
2) Conclusiones clave (máximo 7)
3) Recomendaciones y toma de decisiones (máximo 10) separadas por: Comercial / Operación / Finanzas
4) Riesgos (datos, demanda, mezcla, ejecución) y mitigaciones
5) Próximos 30 días: plan de acción
"""
            ans, err = gemini_generate(prompt, model_preference=model_choice)
            if err:
                st.error("No se pudo generar el análisis. Revisa tu GOOGLE_API_KEY (o GEMINI_API_KEY) y el modelo.")
                st.code(err)
            else:
                st.session_state["last_exec_report_md"] = ans
                st.markdown(ans)

                # Export ejecutivo como Markdown (descarga)
                md_bytes = ans.encode("utf-8")
                st.download_button(
                    "⬇️ Descargar informe ejecutivo (MD)",
                    data=md_bytes,
                    file_name=f"informe_ejecutivo_{anio_sel}.md",
                    mime="text/markdown"
                )


# -------------------------
# TAB 4: FORECAST & RIESGO
# -------------------------
with tab4:
    st.subheader("4) Forecast & Riesgo (nivel MBA)")
    st.caption("Forecast simple + riesgos ejecutivos SIN librerías adicionales. (Puedes mejorar después con modelos avanzados).")

    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña **Cargar Excel**.")
    else:
        df0 = st.session_state["df_final"].copy()
        meses_default = st.session_state.get("meses_analizables", [])
        dff, anio_sel, mes_sel = sidebar_filters(df0, meses_default)

        # Serie mensual
        by_mes = dff.groupby(dff["mes"].astype(str), as_index=True)["actual_kg"].sum()
        by_mes = by_mes.reindex(MESES_ORDEN).fillna(0)

        st.markdown("### Forecast (próximos 3 meses)")
        fc = simple_forecast(by_mes, steps=3)
        st.dataframe(fc, use_container_width=True)

        st.plotly_chart(
            px.bar(fc, x="mes", y="forecast_actual_kg"),
            use_container_width=True
        )

        st.markdown("### Riesgos (resumen ejecutivo)")
        rs = risk_summary(st.session_state["df_final"], mes_sel if mes_sel else meses_default)

        c1, c2, c3 = st.columns(3)
        c1.metric("Riesgo (0–100)", f"{rs['risk_score_0_100']}")
        c2.metric("Cumplimiento periodo", f"{rs['cumplimiento_pct']:.1f}%")
        c3.metric("Ratio líneas bajo presupuesto", f"{rs['under_ratio']*100:.1f}%")

        if rs["meses_cero"]:
            st.warning(f"Meses sin ventas registradas dentro del filtro: {', '.join(rs['meses_cero'])}")

        if rs["conc_vend"]:
            st.info(f"Concentración Vendedor: {rs['conc_vend']['top_vendedor']} = {rs['conc_vend']['top_share_pct']:.1f}% del volumen del periodo.")

        if rs["conc_cli"]:
            st.info(f"Concentración Cliente ({rs['conc_cli']['col']}): {rs['conc_cli']['top_cliente']} = {rs['conc_cli']['top_share_pct']:.1f}% del volumen del periodo.")

        st.markdown("### Export ejecutivo (resumen riesgos)")
        risk_md = f"""# Riesgo Ejecutivo — {anio_sel}

- Risk score (0–100): **{rs['risk_score_0_100']}**
- Cumplimiento del periodo: **{rs['cumplimiento_pct']:.1f}%**
- Meses sin ventas registradas (en filtro): **{', '.join(rs['meses_cero']) if rs['meses_cero'] else 'Ninguno'}**
- Ratio de líneas bajo presupuesto: **{rs['under_ratio']*100:.1f}%**

## Concentración
- Vendedor: {rs['conc_vend'] if rs['conc_vend'] else 'N/A'}
- Cliente: {rs['conc_cli'] if rs['conc_cli'] else 'N/A'}
"""
        st.download_button(
            "⬇️ Descargar (MD)",
            data=risk_md.encode("utf-8"),
            file_name=f"riesgo_ejecutivo_{anio_sel}.md",
            mime="text/markdown"
        )


# -------------------------
# TAB 5: IA TÉCNICA PREVENTA (RAG)
# -------------------------
with tab5:
    st.subheader("5) IA Técnica Preventa (basado en manual PDF)")
    st.caption("Responde SOLO con evidencia del manual. Si falta info, pide los datos faltantes (no inventa).")

    pdf_path = find_manual_pdf()
    if not pdf_path:
        st.warning(
            "No encuentro un PDF del manual.\n\n"
            "✅ Recomendado: súbelo a la repo en `manual_tecnico/`.\n"
            "Ejemplo: `manual_tecnico/Manual_tecnico_preventa.pdf`"
        )
        st.stop()

    st.caption(f"Manual detectado: `{pdf_path}`")

    try:
        manual_text = load_manual_text(pdf_path)
    except Exception as e:
        st.error(f"No pude leer el PDF: {type(e).__name__}: {e}")
        st.stop()

    if not manual_text:
        st.warning("Pude abrir el PDF, pero no pude extraer texto. Si está escaneado, expórtalo como PDF con texto seleccionable.")
        st.stop()

    manual_chunks = chunk_text(manual_text, chunk_size=1300, overlap=180)
    st.success(f"Manual cargado. Chunks: {len(manual_chunks)}")

    model_choice = st.selectbox("Modelo para IA Técnica", DEFAULT_MODEL_CANDIDATES, index=0, key="model_tech")

    # Chat history
    if "chat_tech" not in st.session_state:
        st.session_state["chat_tech"] = []

    # Render history
    for msg in st.session_state["chat_tech"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Ej: Café 500g, VFFS, vida útil 12 meses, zipper, válvula, destino exportación.")
    if pregunta:
        st.session_state["chat_tech"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # RAG context
        context = short_context(pregunta, manual_chunks, k=6, max_chars=6500)

        # Si no encontramos evidencia fuerte, pedimos más datos (sin inventar)
        if not context:
            with st.chat_message("assistant"):
                st.warning(
                    "No encuentro evidencia suficiente en el manual para responder.\n\n"
                    "Para ayudarte, dime por favor:\n"
                    "- Producto y vida útil\n"
                    "- Tipo de empaque (bolsa/bobina) y máquina (VFFS/HFFS)\n"
                    "- Requisitos barrera (O2/H2O/luz)\n"
                    "- Gramaje/peso y dimensiones\n"
                    "- Condiciones de almacenamiento"
                )
            st.session_state["chat_tech"].append({
                "role": "assistant",
                "content": "No encuentro evidencia suficiente en el manual. Por favor comparte: producto/vida útil, tipo de empaque, máquina, barrera, peso/dimensiones y almacenamiento."
            })
        else:
            # Prompt final
            prompt = f"""
Eres ingeniero/a preventa senior en empaque plástico flexible (bolsa y bobina).
Debes responder SOLO usando la EVIDENCIA del manual dentro de CONTEXTO.
Si algo NO está en el contexto, debes pedir los datos faltantes (no inventes).

FORMATO DE SALIDA (obligatorio):
1) Preguntas faltantes (si aplica)
2) Opción A segura (máxima protección)
3) Opción B optimizada costo
4) Riesgos y trade-offs
5) Evidencia usada (citas textuales cortas o referencias al CONTEXTO)

CONTEXTO (EVIDENCIA):
{context}

PREGUNTA DEL CLIENTE:
{pregunta}
"""
            ans, err = gemini_generate(prompt, model_preference=model_choice)
            if err:
                with st.chat_message("assistant"):
                    st.error("Error llamando a Gemini. Revisa tu API key y modelo.")
                    st.code(err)
                st.session_state["chat_tech"].append({"role": "assistant", "content": f"Error Gemini: {err}"})
            else:
                with st.chat_message("assistant"):
                    st.markdown(ans)
                st.session_state["chat_tech"].append({"role": "assistant", "content": ans})


# =========================
# FOOTER: DEBUG OPCIONAL
# =========================
with st.expander("Diagnóstico (opcional)"):
    st.write("Archivos en raíz:", os.listdir("."))
    if os.path.isdir(MANUAL_DIR):
        st.write("Archivos en manual_tecnico/:", os.listdir(MANUAL_DIR))
    st.write("¿API Key detectada?:", "Sí" if _get_google_api_key() else "No")
