# app.py
# Dashboard mensual (Ventas vs Presupuesto KG) + IA Generativa (Gemini) + Asistente técnico preventa (RAG local con PDF)
# ✅ Sin reportlab (evita ModuleNotFoundError). Exporta informe en Markdown + Excel.
# Requisitos sugeridos (requirements.txt):
# streamlit
# pandas
# openpyxl
# plotly
# pypdf2
# google-genai

import os
import re
import io
import math
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader

# Google Gemini (Google AI Studio)
from google import genai
from google.genai import errors as genai_errors

# ===================== CONFIG =====================
st.set_page_config(page_title="Dashboard Mensual + IA Generativa", layout="wide")

APP_TITLE = "📊 Dashboard Mensual + IA Generativa"
MANUAL_DIR = "manual_tecnico"  # carpeta recomendada en repo
MANUAL_FILENAME_HINT = "Manual_tecnico_preventa.pdf"  # nombre sugerido

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

# ===================== HELPERS =====================
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def safe_div(a: float, b: float) -> float:
    if b is None or b == 0:
        return 0.0
    return a / b

def month_index(mes: str) -> int:
    try:
        return MESES_ORDEN.index(mes) + 1
    except Exception:
        return 99

def format_kg(x: float) -> str:
    return f"{x:,.0f}"

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")

# ===================== GEMINI (GOOGLE AI STUDIO) =====================
def get_gemini_key() -> str | None:
    # En Streamlit Cloud: Secrets -> GEMINI_API_KEY = "..."
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = os.environ.get("GEMINI_API_KEY", None)
    return key

def gemini_client():
    api_key = get_gemini_key()
    if not api_key:
        st.error("Falta `GEMINI_API_KEY` en Secrets (Streamlit Cloud).")
        st.stop()
    return genai.Client(api_key=api_key)

def gemini_generate(prompt: str, model_preference: str | None = None) -> str:
    """
    - Evita que se caiga el app si el modelo no existe para tu cuenta.
    - Prueba una lista de modelos comunes.
    """
    client = gemini_client()

    candidates = []
    if model_preference:
        candidates.append(model_preference)

    # Orden sugerido (si uno falla 404, prueba el siguiente)
    candidates += [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
    ]

    last_err = None
    for m in candidates:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=prompt,
            )
            # Algunas respuestas vienen con .text, otras con candidates; .text suele funcionar
            return getattr(resp, "text", "") or str(resp)
        except Exception as e:
            last_err = e
            # si es NOT_FOUND o modelo no disponible, intentamos el siguiente
            continue

    raise RuntimeError(f"No pude generar respuesta con Gemini. Último error: {last_err}")

# ===================== MANUAL PDF (RAG LOCAL) =====================
@st.cache_data(show_spinner=False)
def find_manual_pdf() -> str | None:
    """
    Busca el manual en:
    1) manual_tecnico/*.pdf
    2) raíz del repo *.pdf (por si lo subiste sin carpeta)
    """
    # 1) carpeta manual_tecnico/
    if os.path.isdir(MANUAL_DIR):
        pdfs = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
        pdfs.sort()
        if pdfs:
            return os.path.join(MANUAL_DIR, pdfs[0])

    # 2) raíz
    root_pdfs = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    # prioriza el nombre sugerido
    if MANUAL_FILENAME_HINT in root_pdfs:
        return MANUAL_FILENAME_HINT
    root_pdfs.sort()
    if root_pdfs:
        return root_pdfs[0]

    return None

@st.cache_data(show_spinner=False)
def load_manual_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()

def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
        if j == n:
            break
    return chunks

def simple_retrieve(chunks: list[str], query: str, top_k: int = 6) -> list[str]:
    """
    Retrieval simple por overlap de términos (sin vector DB).
    Suficiente para un PDF corto y evita complejidad.
    """
    q = (query or "").lower()
    tokens = [t for t in re.split(r"[^a-zA-Z0-9áéíóúñü]+", q) if len(t) >= 3]
    if not tokens:
        return chunks[:min(top_k, len(chunks))]

    scored = []
    for c in chunks:
        cl = c.lower()
        score = 0
        for t in tokens:
            score += cl.count(t)
        # pequeño bonus si aparecen palabras típicas
        if "estructura" in cl or "micraje" in cl or "barrera" in cl:
            score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for s, c in scored[:top_k] if s > 0]
    if not top:
        top = [c for _, c in scored[:top_k]]
    return top

# ===================== NORMALIZACIÓN DE EXCEL =====================
def normalizar_ventas(df_ventas: pd.DataFrame) -> pd.DataFrame:
    """
    Espera al menos:
    - ItemCode
    - Columnas mensuales tipo Ene_KG, Feb_KG, ...
    Opcional: SlpName, Nombre de cliente, Código de cliente/proveedor, ItemName
    """
    if "ItemCode" not in df_ventas.columns:
        raise ValueError("Ventas: falta columna obligatoria 'ItemCode'.")

    # columnas opcionales
    opt_cols = [c for c in ["SlpName", "Nombre de cliente", "Código de cliente/proveedor", "ItemName"] if c in df_ventas.columns]

    out = []
    for mes, colkg in MESES_VENTAS.items():
        if colkg not in df_ventas.columns:
            # si tu archivo mensual solo trae algunos meses, permitimos ausencia de columnas:
            # -> se asume 0 para ese mes (pero en análisis YTD se excluirá si no hay data)
            tmp = df_ventas[["ItemCode"] + opt_cols].copy()
            tmp["mes"] = mes
            tmp["actual_kg"] = 0.0
            tmp["col_present"] = False
            out.append(tmp)
            continue

        tmp = df_ventas[["ItemCode"] + opt_cols].copy()
        tmp["mes"] = mes
        tmp["actual_kg"] = _num(df_ventas[colkg])
        tmp["col_present"] = True
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    return long

def normalizar_presupuesto(df_pres: pd.DataFrame) -> pd.DataFrame:
    """
    Espera:
    - ItemCode
    - Columnas ENE..DIC
    """
    if "ItemCode" not in df_pres.columns:
        raise ValueError("Presupuesto: falta columna obligatoria 'ItemCode'.")

    out = []
    for mes, col in MESES_PRES.items():
        if col not in df_pres.columns:
            tmp = df_pres[["ItemCode"]].copy()
            tmp["mes"] = mes
            tmp["budget_kg"] = 0.0
            tmp["col_present"] = False
            out.append(tmp)
            continue

        tmp = df_pres[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["budget_kg"] = _num(df_pres[col])
        tmp["col_present"] = True
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    return long

def build_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fact table por ItemCode-mes:
    - actual_kg
    - budget_kg
    - var_kg
    - cumpl_pct
    """
    merged = df["ventas"].merge(df["pres"], on=["ItemCode", "mes"], how="left", suffixes=("_act", "_bud"))
    merged["budget_kg"] = merged["budget_kg"].fillna(0.0)
    merged["var_kg"] = merged["actual_kg"] - merged["budget_kg"]
    merged["cumpl_pct"] = merged.apply(lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0.0, axis=1)

    # define "hay_data" real: mes con columna presente en ventas + actual_kg > 0
    merged["hay_data"] = merged["col_present_act"].fillna(False) & (merged["actual_kg"] > 0)

    return merged

# ===================== KPI / FORECAST / RIESGO =====================
def kpis_ytd(df_fact: pd.DataFrame, incluir_meses_sin_ventas: bool) -> dict:
    if incluir_meses_sin_ventas:
        df_use = df_fact.copy()
    else:
        # Solo meses donde hay ventas reales >0 (YTD real), evita “castigar” meses aún no cargados
        df_use = df_fact[df_fact["hay_data"]].copy()

    actual = float(df_use["actual_kg"].sum())
    budget = float(df_use["budget_kg"].sum())
    var = actual - budget
    cumpl = safe_div(actual, budget) * 100
    meses_con_data = int(df_use["mes"].nunique()) if len(df_use) else 0

    return {
        "actual": actual,
        "budget": budget,
        "var": var,
        "cumpl": cumpl,
        "meses_con_data": meses_con_data,
    }

def forecast_simple(df_fact: pd.DataFrame) -> dict:
    """
    Forecast anual simple:
    - Run-rate promedio de meses con data * 12
    - Run-rate último mes con data * 12
    """
    df_data = df_fact[df_fact["hay_data"]].copy()
    if df_data.empty:
        return {"status": "sin_data"}

    by_mes = df_data.groupby("mes", as_index=False)["actual_kg"].sum()
    by_mes["mes_idx"] = by_mes["mes"].astype(str).apply(month_index)
    by_mes = by_mes.sort_values("mes_idx")

    meses = len(by_mes)
    total = float(by_mes["actual_kg"].sum())
    avg = total / meses if meses > 0 else 0.0

    last = float(by_mes.iloc[-1]["actual_kg"])
    fc_avg = avg * 12
    fc_last = last * 12

    return {
        "status": "ok",
        "meses": meses,
        "ytd_total": total,
        "avg_run_rate": avg,
        "last_month": last,
        "forecast_avg": fc_avg,
        "forecast_last": fc_last,
        "by_mes": by_mes,
    }

def riesgo_heuristico(df_fact: pd.DataFrame) -> dict:
    """
    Riesgo (heurístico, sin inventar):
    - Data quality: pocos meses con data
    - Concentración: % del total en top 1 mes
    - Gap: cumplimiento YTD
    """
    df_data = df_fact[df_fact["hay_data"]].copy()
    if df_data.empty:
        return {"nivel": "ALTO", "razones": ["No hay meses con ventas cargadas (data faltante)."]}

    by_mes = df_data.groupby("mes", as_index=False)["actual_kg"].sum()
    total = float(by_mes["actual_kg"].sum())
    top1 = float(by_mes["actual_kg"].max())
    conc = safe_div(top1, total) * 100

    meses = int(by_mes["mes"].nunique())
    # KPI YTD (solo data)
    k = kpis_ytd(df_fact, incluir_meses_sin_ventas=False)
    cumpl = k["cumpl"]

    score = 0
    razones = []

    if meses <= 2:
        score += 2
        razones.append("Pocos meses con data real cargada (<=2). Riesgo de interpretación/estacionalidad.")
    elif meses <= 5:
        score += 1
        razones.append("Aún hay pocos meses con data real (<=5).")

    if conc >= 65:
        score += 2
        razones.append(f"Alta concentración: el mes pico representa ~{conc:.0f}% del volumen cargado.")
    elif conc >= 45:
        score += 1
        razones.append(f"Concentración media: el mes pico representa ~{conc:.0f}% del volumen cargado.")

    if cumpl < 80:
        score += 2
        razones.append(f"Cumplimiento YTD bajo (<80%): {cumpl:.1f}%.")
    elif cumpl < 95:
        score += 1
        razones.append(f"Cumplimiento YTD medio (<95%): {cumpl:.1f}%.")

    if score >= 4:
        nivel = "ALTO"
    elif score >= 2:
        nivel = "MEDIO"
    else:
        nivel = "BAJO"

    return {"nivel": nivel, "razones": razones}

# ===================== EXPORTACIÓN (MD + XLSX) =====================
def build_exec_markdown(kpis_scope: dict, forecast: dict, riesgo: dict, notas: str, incluir_meses_sin_ventas: bool) -> str:
    scope = "Año completo (incluye meses sin ventas)" if incluir_meses_sin_ventas else "YTD real (solo meses con ventas cargadas)"
    md = []
    md.append(f"# Informe Ejecutivo — Ventas vs Presupuesto (KG)\n")
    md.append(f"- **Fecha/hora:** {now_str()}")
    md.append(f"- **Alcance:** {scope}\n")

    md.append("## KPIs\n")
    md.append(f"- **Actual (KG):** {format_kg(kpis_scope['actual'])}")
    md.append(f"- **Budget (KG):** {format_kg(kpis_scope['budget'])}")
    md.append(f"- **Varianza (KG):** {format_kg(kpis_scope['var'])}")
    md.append(f"- **% Cumplimiento:** {kpis_scope['cumpl']:.1f}%")
    md.append(f"- **Meses con data (ventas >0):** {kpis_scope['meses_con_data']}\n")

    md.append("## Forecast (simple)\n")
    if forecast.get("status") != "ok":
        md.append("- Sin data suficiente para forecast.\n")
    else:
        md.append(f"- **YTD total (KG):** {format_kg(forecast['ytd_total'])} en {forecast['meses']} mes(es) con data")
        md.append(f"- **Run-rate promedio mensual:** {format_kg(forecast['avg_run_rate'])} KG/mes")
        md.append(f"- **Forecast anual (promedio):** {format_kg(forecast['forecast_avg'])} KG")
        md.append(f"- **Run-rate último mes:** {format_kg(forecast['last_month'])} KG/mes")
        md.append(f"- **Forecast anual (último mes):** {format_kg(forecast['forecast_last'])} KG\n")

    md.append("## Riesgo (heurístico)\n")
    md.append(f"- **Nivel:** {riesgo.get('nivel','N/A')}")
    razones = riesgo.get("razones", [])
    if razones:
        md.append("- **Razones:**")
        for r in razones:
            md.append(f"  - {r}")
    md.append("")

    if notas:
        md.append("## Notas / Decisiones\n")
        md.append(notas.strip())
        md.append("")

    return "\n".join(md)

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "fact") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# ===================== UI =====================
st.title(APP_TITLE)

# Sidebar: configuración IA
with st.sidebar:
    st.header("⚙️ Configuración IA")
    st.caption("Gemini desde Google AI Studio")
    model_choice = st.text_input("Modelo (opcional)", value="gemini-1.5-flash")
    st.caption("Secrets requeridos: `GEMINI_API_KEY`")

    st.divider()
    st.header("📌 Alcance de KPIs")
    incluir_meses_sin_ventas = st.checkbox(
        "Incluir meses sin ventas (año completo)",
        value=False,
        help="Si está apagado, calcula YTD real (solo meses con ventas cargadas) para que no te baje el % cuando aún no has actualizado meses futuros."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Cargar Excel",
    "2) Dashboard",
    "3) IA del Dashboard",
    "4) Forecast & Riesgo",
    "5) IA Técnica Preventa"
])

# -------- TAB 1: CARGA --------
with tab1:
    st.subheader("Carga mensual de data (Excel)")
    colA, colB = st.columns(2)

    with colA:
        ventas_file = st.file_uploader("Reporte Ventas (.xlsx)", type=["xlsx"], key="ventas")
    with colB:
        pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"], key="pres")

    if st.button("Procesar", type="primary"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos (Ventas y Presupuesto).")
        else:
            try:
                dfv = pd.read_excel(ventas_file)
                dfp = pd.read_excel(pres_file)

                ventas_long = normalizar_ventas(dfv)
                pres_long = normalizar_presupuesto(dfp)

                df_fact = build_fact_table({"ventas": ventas_long, "pres": pres_long})

                st.session_state["df_fact"] = df_fact
                st.success("✅ Datos procesados.")
                st.caption("Vista previa (fact table):")
                st.dataframe(df_fact.head(20), use_container_width=True)
            except Exception as e:
                st.exception(e)

# -------- TAB 2: DASHBOARD --------
with tab2:
    st.subheader("Ventas vs Presupuesto (KG)")
    if "df_fact" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df = st.session_state["df_fact"].copy()

        # KPIs con alcance definido por checkbox
        k = kpis_ytd(df, incluir_meses_sin_ventas=incluir_meses_sin_ventas)
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        c1.metric("Actual (KG)", format_kg(k["actual"]))
        c2.metric("Budget (KG)", format_kg(k["budget"]))
        c3.metric("Varianza (KG)", format_kg(k["var"]))
        c4.metric("% Cumplimiento", f"{k['cumpl']:.1f}%")
        c5.metric("Meses con data", f"{k['meses_con_data']}")

        st.markdown("### Tendencia mensual (Actual vs Budget)")
        if incluir_meses_sin_ventas:
            df_use = df.copy()
        else:
            df_use = df[df["hay_data"]].copy()

        by_mes = df_use.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum()
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        if by_mes.empty:
            st.info("Aún no hay meses con ventas > 0 para graficar.")
        else:
            st.plotly_chart(
                px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True),
                use_container_width=True
            )

        st.markdown("### Top SKUs por varianza (KG) (según alcance)")
        top = (
            df_use.groupby(["ItemCode"], as_index=False)[["actual_kg", "budget_kg", "var_kg"]]
            .sum()
            .sort_values("var_kg", ascending=True)
        )
        st.dataframe(top.head(25), use_container_width=True)

# -------- TAB 3: IA DEL DASHBOARD --------
with tab3:
    st.subheader("IA del Dashboard (análisis ejecutivo)")
    if "df_fact" not in st.session_state:
        st.warning("Carga datos primero.")
    else:
        df = st.session_state["df_fact"].copy()
        df_use = df.copy() if incluir_meses_sin_ventas else df[df["hay_data"]].copy()

        by_mes = (
            df_use.groupby("mes", as_index=False)[["actual_kg", "budget_kg", "var_kg"]]
            .sum()
        )
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        # Texto de datos (solo lo necesario)
        resumen = by_mes.to_string(index=False)
        k = kpis_ytd(df, incluir_meses_sin_ventas=incluir_meses_sin_ventas)

        st.caption("Genera un informe (sin inventar cifras) con base en los datos agregados por mes.")
        if st.button("Generar análisis con IA", type="primary"):
            try:
                scope = "Año completo (incluye meses sin ventas)" if incluir_meses_sin_ventas else "YTD real (solo meses con ventas cargadas)"
                prompt = f"""
Eres un Asistente Senior de Comercialización y Finanzas Industriales (empaque plástico flexible).

REGLAS:
- Usa SOLO los datos proporcionados.
- No inventes cifras.
- Si detectas ausencia de meses (por falta de carga), no lo asumas como “cero ventas”; indícalo como “data no cargada” si aplica.

ALCANCE KPI: {scope}

DATOS POR MES (KG):
{resumen}

KPIs:
- Actual: {k['actual']}
- Budget: {k['budget']}
- Varianza: {k['var']}
- Cumplimiento %: {k['cumpl']}
- Meses con data: {k['meses_con_data']}

FORMATO DE SALIDA:
1) Resumen ejecutivo (5 bullets)
2) Hallazgos clave (3–6 bullets)
3) Recomendaciones comerciales (3–5 bullets, accionables)
4) Riesgos y supuestos críticos (3–6 bullets)
"""
                ans = gemini_generate(prompt, model_preference=model_choice.strip() or None)
                st.markdown(ans)
                st.session_state["last_ai_dashboard"] = ans
            except genai_errors.ClientError as e:
                st.error(f"Error llamando a Gemini: {e}")
            except Exception as e:
                st.error("No se pudo generar el análisis. Revisa tu `GEMINI_API_KEY` y el modelo.")
                st.exception(e)

# -------- TAB 4: FORECAST & RIESGO + EXPORT --------
with tab4:
    st.subheader("Forecast & Riesgo (MBA-level, simple y explicable)")
    if "df_fact" not in st.session_state:
        st.warning("Carga datos primero.")
    else:
        df = st.session_state["df_fact"].copy()

        fc = forecast_simple(df)
        rg = riesgo_heuristico(df)
        k_scope = kpis_ytd(df, incluir_meses_sin_ventas=incluir_meses_sin_ventas)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Forecast (run-rate)")
            if fc.get("status") != "ok":
                st.info("Sin data suficiente para forecast (necesitas al menos 1 mes con ventas > 0).")
            else:
                st.write(f"- Meses con data: **{fc['meses']}**")
                st.write(f"- YTD total: **{format_kg(fc['ytd_total'])} KG**")
                st.write(f"- Run-rate promedio: **{format_kg(fc['avg_run_rate'])} KG/mes**")
                st.write(f"- Forecast anual (promedio): **{format_kg(fc['forecast_avg'])} KG**")
                st.write(f"- Run-rate último mes: **{format_kg(fc['last_month'])} KG/mes**")
                st.write(f"- Forecast anual (último mes): **{format_kg(fc['forecast_last'])} KG**")

        with c2:
            st.markdown("#### Riesgo (heurístico)")
            st.write(f"**Nivel:** {rg.get('nivel','N/A')}")
            for r in rg.get("razones", []):
                st.write(f"- {r}")

        st.divider()
        st.subheader("Exportación ejecutiva (sin PDF)")
        notas = st.text_area(
            "Notas / Decisiones (opcional)",
            placeholder="Ej: Acciones del mes, acuerdos con gerencia, supuestos del forecast, etc.",
            height=120
        )

        md_report = build_exec_markdown(
            kpis_scope=k_scope,
            forecast=fc,
            riesgo=rg,
            notas=notas,
            incluir_meses_sin_ventas=incluir_meses_sin_ventas
        )

        colx, coly = st.columns(2)
        with colx:
            st.download_button(
                "⬇️ Descargar Informe (Markdown .md)",
                data=md_report.encode("utf-8"),
                file_name="informe_ejecutivo_ventas_presupuesto.md",
                mime="text/markdown"
            )
        with coly:
            excel_bytes = df_to_excel_bytes(st.session_state["df_fact"], sheet_name="fact_ventas_budget")
            st.download_button(
                "⬇️ Descargar Data (Excel .xlsx)",
                data=excel_bytes,
                file_name="fact_ventas_budget.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # (Opcional) IA para “medir impacto simulado”
        st.markdown("#### Medición de impacto (simulada con IA)")
        st.caption("Ejemplo MBA: ¿qué pasa si subimos +10% el run-rate desde el próximo mes? (IA explica, no inventa datos base).")
        if st.button("Simular impacto con IA"):
            try:
                fc_txt = "SIN FORECAST" if fc.get("status") != "ok" else (
                    f"YTD={fc['ytd_total']}, meses={fc['meses']}, avg_run_rate={fc['avg_run_rate']}, last={fc['last_month']}"
                )
                prompt = f"""
Eres consultor ejecutivo (MBA) para una empresa de empaque flexible.

Base (no inventar):
- KPIs: actual={k_scope['actual']}, budget={k_scope['budget']}, var={k_scope['var']}, cumpl={k_scope['cumpl']}
- Forecast inputs: {fc_txt}
- Riesgo nivel: {rg.get('nivel')}

Tarea:
1) Define 2 escenarios (Base y +10% run-rate desde próximo mes)
2) Explica impacto cualitativo en cumplimiento anual vs presupuesto (sin inventar presupuesto anual si no está)
3) Recomienda 3 acciones comerciales/operativas para capturar el upside
4) Lista supuestos críticos y riesgos

Formato: bullets claros, estilo presentación ejecutiva.
"""
                ans = gemini_generate(prompt, model_preference=model_choice.strip() or None)
                st.markdown(ans)
            except Exception as e:
                st.error("No pude simular con IA. Revisa API Key/modelo.")
                st.exception(e)

# -------- TAB 5: IA TÉCNICA PREVENTA (RAG LOCAL) --------
with tab5:
    st.subheader("IA Técnica Preventa (basada en tu manual PDF)")
    st.caption("Responde SOLO con evidencia del manual. Si falta info, pide datos. No inventa.")

    pdf_path = find_manual_pdf()
    if not pdf_path:
        st.warning(
            "No encuentro ningún PDF del manual.\n\n"
            "✅ Solución recomendada:\n"
            "1) Crea carpeta `manual_tecnico/` en tu repo.\n"
            "2) Sube tu PDF ahí (ej: `manual_tecnico/Manual_tecnico_preventa.pdf`).\n"
            "3) Redeploy en Streamlit Cloud.\n"
        )
        st.stop()

    st.success(f"Manual detectado: `{pdf_path}`")

    try:
        manual_text = load_manual_text(pdf_path)
    except Exception as e:
        st.error("No pude abrir el PDF.")
        st.exception(e)
        st.stop()

    if not manual_text:
        st.warning("Pude abrir el PDF, pero no pude extraer texto (posible PDF escaneado).")
        st.stop()

    chunks = chunk_text(manual_text, chunk_chars=1200, overlap=150)

    # Chat history
    if "chat_tecnico" not in st.session_state:
        st.session_state["chat_tecnico"] = []

    for msg in st.session_state["chat_tecnico"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Ej: Café 500g, VFFS, vida útil 12 meses, ¿estructura y micraje?")
    if pregunta:
        st.session_state["chat_tecnico"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        # retrieval
        top_chunks = simple_retrieve(chunks, pregunta, top_k=6)
        contexto = "\n\n---\n\n".join(top_chunks)

        prompt = f"""
Eres ingeniero preventa experto en empaque plástico flexible (bolsa/bobina).

REGLAS ESTRICTAS:
- Responde SOLO usando evidencia explícita del CONTEXTO (manual).
- Si no hay evidencia suficiente, NO inventes. Pide datos faltantes.
- Si el manual no cubre algo, dilo claramente.

FORMATO DE SALIDA:
1) Datos faltantes (si aplica) — bullets
2) Opción A segura (máxima protección) — estructura / micraje / barrera / notas de proceso
3) Opción B optimizada costo — estructura / micraje / barrera / notas de proceso
4) Riesgos y supuestos críticos — bullets
5) Evidencia (citas) — pega 2–6 fragmentos exactos del CONTEXTO que sustentan tus puntos (corto)

CONTEXTO (manual):
{contexto}

PREGUNTA:
{pregunta}
"""
        try:
            respuesta = gemini_generate(prompt, model_preference=model_choice.strip() or None)
            with st.chat_message("assistant"):
                st.markdown(respuesta)
            st.session_state["chat_tecnico"].append({"role": "assistant", "content": respuesta})
        except Exception as e:
            with st.chat_message("assistant"):
                st.error("No pude responder con IA. Revisa `GEMINI_API_KEY` y el modelo.")
                st.exception(e)

# ===================== DEBUG (descomenta si lo necesitas) =====================
# st.write("Contenido raíz:", os.listdir("."))
# if os.path.isdir(MANUAL_DIR):
#     st.write("Contenido manual_tecnico:", os.listdir(MANUAL_DIR))
