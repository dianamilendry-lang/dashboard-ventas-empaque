import os
import re
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader
from google import genai

# ========= PDF export (reportlab) =========
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ===================== CONFIG =====================
st.set_page_config(page_title="Dashboard Mensual + IA Generativa", layout="wide")

MESES_ORDEN = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
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

# PDF del manual (según tu repo, lo tienes en la raíz)
MANUAL_PATH_ROOT = "Manual_tecnico_preventa.pdf"
MANUAL_DIR = "manual_tecnico"


# ===================== GEMINI =====================
def gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error("Falta GEMINI_API_KEY en Secrets (Streamlit Cloud).")
        st.stop()
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False)
def pick_gemini_model():
    """
    Elige un modelo disponible para tu API Key.
    Evita modelos retirados para usuarios nuevos.
    """
    client = gemini_client()
    models = client.models.list()

    preferred = [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    available = []
    for m in models:
        name = getattr(m, "name", "")
        if not name:
            continue
        model_id = name.replace("models/", "")

        actions = getattr(m, "supported_actions", None) or getattr(m, "supportedActions", None)
        if actions and ("generateContent" not in actions):
            continue

        available.append(model_id)

    for p in preferred:
        if p in available:
            return p

    # fallback razonable
    for mid in available:
        if "flash" in mid or "pro" in mid:
            return mid

    return available[0] if available else None


def gemini_generate(prompt: str) -> str:
    try:
        client = gemini_client()
        model_id = pick_gemini_model()
        if not model_id:
            return "❌ No encontré modelos disponibles con generateContent para esta API key."
        resp = client.models.generate_content(model=model_id, contents=prompt)
        return resp.text or "(Sin respuesta)"
    except Exception as e:
        return f"❌ Error llamando a Gemini: {type(e).__name__} — {e}"


# ===================== MANUAL (RAG simple) =====================
@st.cache_data(show_spinner=False)
def find_manual_pdf() -> str | None:
    # 1) raíz
    if os.path.exists(MANUAL_PATH_ROOT):
        return MANUAL_PATH_ROOT
    # 2) carpeta
    if os.path.isdir(MANUAL_DIR):
        pdfs = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
        if pdfs:
            pdfs.sort()
            return os.path.join(MANUAL_DIR, pdfs[0])
    return None


@st.cache_data(show_spinner=False)
def load_manual_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    txt = []
    for p in reader.pages:
        txt.append(p.extract_text() or "")
    return "\n".join(txt).strip()


def tokenize(text: str) -> set:
    text = re.sub(r"[^a-zA-Z0-9áéíóúñ ]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return set([t for t in text.split(" ") if len(t) >= 3])


@st.cache_data(show_spinner=False)
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += (chunk_size - overlap)
    return chunks


def retrieve_chunks(chunks: list[str], query: str, top_k: int = 4) -> list[str]:
    q = tokenize(query)
    scored = []
    for c in chunks:
        score = len(q.intersection(tokenize(c)))
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ===================== DATA (mensual, Excel) =====================
def normalizar_ventas(df: pd.DataFrame) -> pd.DataFrame:
    if "ItemCode" not in df.columns:
        raise ValueError("Ventas: falta columna 'ItemCode'.")
    out = []
    for mes, col in MESES_VENTAS.items():
        if col not in df.columns:
            raise ValueError(f"Ventas: falta columna mensual '{col}'.")
        tmp = df[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["actual_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out.append(tmp)
    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    return long


def normalizar_pres(df: pd.DataFrame) -> pd.DataFrame:
    if "ItemCode" not in df.columns:
        raise ValueError("Presupuesto: falta columna 'ItemCode'.")
    out = []
    for mes, col in MESES_PRES.items():
        if col not in df.columns:
            raise ValueError(f"Presupuesto: falta columna mensual '{col}'.")
        tmp = df[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["budget_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out.append(tmp)
    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    return long


def build_df_final(ventas_long: pd.DataFrame, pres_long: pd.DataFrame) -> pd.DataFrame:
    df = ventas_long.merge(pres_long, on=["ItemCode", "mes"], how="left")
    df["budget_kg"] = df["budget_kg"].fillna(0)
    df["var_kg"] = df["actual_kg"] - df["budget_kg"]
    df["cumpl_pct"] = (df["actual_kg"] / df["budget_kg"]).replace([float("inf")], 0).fillna(0) * 100
    return df


def meses_reportados(df: pd.DataFrame) -> list[str]:
    by_mes = df.groupby("mes", as_index=False)["actual_kg"].sum()
    by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
    by_mes = by_mes.sort_values("mes")
    return [str(m) for m in by_mes[by_mes["actual_kg"] > 0]["mes"].tolist()]


def ytd_kpis(df: pd.DataFrame) -> dict:
    """
    KPIs sobre meses reportados (meses con actual_kg > 0), para no distorsionar % si solo hay 2 meses cargados.
    """
    by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum()
    by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
    by_mes = by_mes.sort_values("mes")
    act = by_mes[by_mes["actual_kg"] > 0].copy()
    if act.empty:
        return {"actual": 0.0, "budget": 0.0, "var": 0.0, "pct": 0.0, "meses": []}
    actual = float(act["actual_kg"].sum())
    budget = float(act["budget_kg"].sum())
    var = actual - budget
    pct = (actual / budget * 100) if budget > 0 else 0.0
    meses = [str(m) for m in act["mes"].tolist()]
    return {"actual": actual, "budget": budget, "var": var, "pct": pct, "meses": meses}


def full_year_kpis(df: pd.DataFrame) -> dict:
    actual = float(df["actual_kg"].sum())
    budget = float(df["budget_kg"].sum())
    var = actual - budget
    pct = (actual / budget * 100) if budget > 0 else 0.0
    return {"actual": actual, "budget": budget, "var": var, "pct": pct}


def forecast_scenarios(df: pd.DataFrame) -> dict | None:
    """
    Forecast anual simple (3 escenarios) basado en meses reportados.
    Devuelve dict con cierre estimado anual en KG para conservador/tendencial/optimista.
    """
    by_mes = df.groupby("mes", as_index=False)["actual_kg"].sum()
    by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
    by_mes = by_mes.sort_values("mes")
    by_mes = by_mes[by_mes["actual_kg"] > 0].copy()

    if len(by_mes) < 2:
        return None

    meses_cargados = len(by_mes)
    meses_restantes = 12 - meses_cargados

    # crecimiento promedio (% change medio)
    growth = float(by_mes["actual_kg"].pct_change().replace([float("inf"), -float("inf")], 0).fillna(0).mean())
    last = float(by_mes["actual_kg"].iloc[-1])

    ytd_actual = float(by_mes["actual_kg"].sum())

    # Tendencial: proyecta usando last*(1+growth)^t como aproximación de "run" futuro mensual promedio
    # Simplificación ejecutiva: future_month_avg ~ last*(1+growth)
    future_month_avg = max(0.0, last * (1 + growth))
    tend_future = future_month_avg * meses_restantes
    opt_future = (future_month_avg * 1.10) * meses_restantes
    cons_future = (future_month_avg * 0.90) * meses_restantes

    return {
        "meses_cargados": meses_cargados,
        "growth_prom": growth,
        "ytd_actual": ytd_actual,
        "conservador": ytd_actual + cons_future,
        "tendencial": ytd_actual + tend_future,
        "optimista": ytd_actual + opt_future,
    }


def risk_score(df: pd.DataFrame, ytd: dict) -> dict:
    """
    Score de riesgo simple (0-6):
    +2 si faltan >6 meses por reportar
    +2 si volatilidad mensual alta
    +2 si cumplimiento YTD < 90%
    """
    meses_cargados = len(ytd.get("meses", []))
    missing = 12 - meses_cargados
    pct_missing = missing / 12

    by_mes = df.groupby("mes")["actual_kg"].sum()
    vol = float(by_mes.std()) if len(by_mes) > 1 else 0.0

    score = 0
    if pct_missing > 0.5:
        score += 2
    if vol > 50000:  # umbral genérico; ajústalo si quieres
        score += 2
    if ytd.get("pct", 0.0) < 90:
        score += 2

    if score <= 2:
        level = "🟢 Bajo"
    elif score <= 4:
        level = "🟡 Medio"
    else:
        level = "🔴 Alto"

    return {
        "score": score,
        "level": level,
        "missing_months": missing,
        "volatilidad": vol,
    }


# ===================== PDF REPORT =====================
def make_executive_pdf(
    file_path: str,
    titulo: str,
    kpi_ytd: dict,
    kpi_fy: dict,
    forecast: dict | None,
    riesgo: dict,
    analisis_ia: str,
    tabla_mensual: pd.DataFrame
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_path, pagesize=LETTER)
    elements = []

    elements.append(Paragraph(titulo, styles["Heading1"]))
    elements.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("KPIs (Meses reportados - YTD)", styles["Heading2"]))
    ytd_tbl = [
        ["Actual (KG)", f"{kpi_ytd['actual']:,.0f}"],
        ["Budget (KG)", f"{kpi_ytd['budget']:,.0f}"],
        ["Varianza (KG)", f"{kpi_ytd['var']:,.0f}"],
        ["% Cumplimiento", f"{kpi_ytd['pct']:.1f}%"],
        ["Meses reportados", ", ".join(kpi_ytd["meses"]) if kpi_ytd["meses"] else "N/A"],
    ]
    t = Table(ytd_tbl, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("KPIs (Full Year - referencia)", styles["Heading2"]))
    fy_tbl = [
        ["Actual FY (KG)", f"{kpi_fy['actual']:,.0f}"],
        ["Budget FY (KG)", f"{kpi_fy['budget']:,.0f}"],
        ["Varianza FY (KG)", f"{kpi_fy['var']:,.0f}"],
        ["% Cumplimiento FY", f"{kpi_fy['pct']:.1f}%"],
    ]
    t2 = Table(fy_tbl, hAlign="LEFT")
    t2.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Forecast (Cierre anual estimado)", styles["Heading2"]))
    if forecast:
        f_tbl = [
            ["Escenario", "Cierre estimado (KG)"],
            ["Conservador", f"{forecast['conservador']:,.0f}"],
            ["Tendencial", f"{forecast['tendencial']:,.0f}"],
            ["Optimista", f"{forecast['optimista']:,.0f}"],
        ]
        tf = Table(f_tbl, hAlign="LEFT")
        tf.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        elements.append(tf)
    else:
        elements.append(Paragraph("No hay suficientes meses para proyectar (se requieren al menos 2 meses reportados).", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Riesgo Comercial (Score)", styles["Heading2"]))
    r_tbl = [
        ["Nivel", riesgo["level"]],
        ["Score (0-6)", str(riesgo["score"])],
        ["Meses sin datos", str(riesgo["missing_months"])],
        ["Volatilidad (std KG)", f"{riesgo['volatilidad']:,.0f}"],
    ]
    tr = Table(r_tbl, hAlign="LEFT")
    tr.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    elements.append(tr)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Análisis Ejecutivo (IA Generativa)", styles["Heading2"]))
    # Evita PDFs muy largos: truncar
    max_chars = 5000
    ia_text = (analisis_ia or "").strip()
    if len(ia_text) > max_chars:
        ia_text = ia_text[:max_chars] + "…"
    elements.append(Paragraph(ia_text.replace("\n", "<br/>"), styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Tabla mensual (resumen)", styles["Heading2"]))
    tbl = tabla_mensual.copy()
    # to list
    rows = [["Mes", "Actual (KG)", "Budget (KG)", "Var (KG)", "% Cumpl"]]
    for _, r in tbl.iterrows():
        rows.append([str(r["mes"]), f"{r['actual_kg']:,.0f}", f"{r['budget_kg']:,.0f}", f"{r['var_kg']:,.0f}", f"{r['cumpl_pct']:.1f}%"])
    tt = Table(rows, hAlign="LEFT")
    tt.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    elements.append(tt)

    doc.build(elements)


# ===================== UI =====================
st.title("📊 Dashboard Mensual + IA Generativa (MBA)")

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Cargar Excel",
    "2) Dashboard",
    "3) IA Ejecutiva + Forecast + Riesgo + PDF",
    "4) IA Técnica Preventa (RAG Manual)"
])

# -------- TAB 1 --------
with tab1:
    st.subheader("Carga de datos mensuales (Excel)")
    ventas_file = st.file_uploader("Reporte Ventas mensual (.xlsx)", type=["xlsx"], key="ventas")
    pres_file = st.file_uploader("Presupuesto mensual (.xlsx)", type=["xlsx"], key="pres")

    if st.button("Procesar archivos"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos (Ventas y Presupuesto).")
        else:
            try:
                dfv = pd.read_excel(ventas_file)
                dfp = pd.read_excel(pres_file)

                ventas_long = normalizar_ventas(dfv)
                pres_long = normalizar_pres(dfp)

                df_final = build_df_final(ventas_long, pres_long)
                st.session_state["df_final"] = df_final

                st.success("✅ Datos procesados. Ve a Dashboard e IA Ejecutiva.")
                with st.expander("Ver muestra (df_final)"):
                    st.dataframe(df_final.head(20), use_container_width=True)
            except Exception as e:
                st.exception(e)

# -------- TAB 2 --------
with tab2:
    st.subheader("Dashboard (cumplimiento vs presupuesto) — KPI sobre meses reportados")

    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df = st.session_state["df_final"].copy()

        # KPIs (YTD meses reportados)
        ytd = ytd_kpis(df)
        fy = full_year_kpis(df)

        st.caption(f"Meses reportados (ventas > 0): {', '.join(ytd['meses']) if ytd['meses'] else 'Ninguno'}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual YTD (KG)", f"{ytd['actual']:,.0f}")
        c2.metric("Budget YTD (KG)", f"{ytd['budget']:,.0f}")
        c3.metric("Varianza YTD (KG)", f"{ytd['var']:,.0f}")
        c4.metric("% Cumplimiento YTD", f"{ytd['pct']:.1f}%")

        with st.expander("Referencia Full Year (FY)"):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Actual FY (KG)", f"{fy['actual']:,.0f}")
            d2.metric("Budget FY (KG)", f"{fy['budget']:,.0f}")
            d3.metric("Var FY (KG)", f"{fy['var']:,.0f}")
            d4.metric("% FY", f"{fy['pct']:.1f}%")

        # Tendencia (solo meses reportados)
        by_mes = (
            df.groupby("mes", as_index=False)[["actual_kg","budget_kg"]].sum()
        )
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")
        by_mes = by_mes[by_mes["actual_kg"] > 0].copy()

        st.markdown("### Tendencia mensual (meses reportados)")
        if by_mes.empty:
            st.info("Aún no hay meses con ventas > 0.")
        else:
            st.plotly_chart(
                px.line(by_mes, x="mes", y=["actual_kg","budget_kg"], markers=True),
                use_container_width=True
            )

# -------- TAB 3 --------
with tab3:
    st.subheader("IA Ejecutiva + Forecast + Riesgo + Exportación PDF")

    if "df_final" not in st.session_state:
        st.warning("Carga datos primero en la pestaña 1.")
    else:
        df = st.session_state["df_final"].copy()
        ytd = ytd_kpis(df)
        fy = full_year_kpis(df)
        fc = forecast_scenarios(df)
        riesgo = risk_score(df, ytd)

        st.caption(f"Modelo Gemini en uso: {pick_gemini_model()}")

        # Tabla mensual resumen (para IA y PDF)
        resumen_mes = df.groupby("mes", as_index=False)[["actual_kg","budget_kg","var_kg"]].sum()
        resumen_mes["cumpl_pct"] = (resumen_mes["actual_kg"]/resumen_mes["budget_kg"]).replace([float("inf")],0).fillna(0)*100
        resumen_mes["mes"] = pd.Categorical(resumen_mes["mes"], categories=MESES_ORDEN, ordered=True)
        resumen_mes = resumen_mes.sort_values("mes")

        c1, c2, c3 = st.columns(3)
        c1.metric("Riesgo", riesgo["level"])
        if fc:
            c2.metric("Forecast Tendencial (KG)", f"{fc['tendencial']:,.0f}")
        else:
            c2.metric("Forecast", "N/A (mín. 2 meses)")
        c3.metric("Cumplimiento YTD", f"{ytd['pct']:.1f}%")

        with st.expander("Ver resumen mensual"):
            st.dataframe(resumen_mes, use_container_width=True)

        st.markdown("### 🔮 Forecast (3 escenarios)")
        if fc:
            f1, f2, f3 = st.columns(3)
            f1.metric("Conservador", f"{fc['conservador']:,.0f} KG")
            f2.metric("Tendencial", f"{fc['tendencial']:,.0f} KG")
            f3.metric("Optimista", f"{fc['optimista']:,.0f} KG")
            st.caption(f"Crecimiento promedio mensual (aprox): {fc['growth_prom']*100:.1f}% | Meses cargados: {fc['meses_cargados']}")
        else:
            st.info("Se requieren al menos 2 meses reportados para forecast.")

        st.markdown("### ⚠️ Score de riesgo")
        st.write(f"Nivel: {riesgo['level']} | Score: {riesgo['score']}/6 | Meses sin datos: {riesgo['missing_months']} | Volatilidad (std): {riesgo['volatilidad']:,.0f} KG")

        st.markdown("### 🤖 Análisis ejecutivo con IA (solo con números del dashboard)")
        if "analisis_ia" not in st.session_state:
            st.session_state["analisis_ia"] = ""

        if st.button("Generar análisis ejecutivo (IA)"):
            prompt = f"""
Eres consultor ejecutivo (industria empaque flexible).
Reglas:
- Usa SOLO los números proporcionados.
- No inventes cifras.
- Si falta información para una recomendación, pide el dato faltante.
- Enfócate en acciones comerciales/operativas.

KPIs YTD (meses reportados):
- Actual: {ytd['actual']}
- Budget: {ytd['budget']}
- Var: {ytd['var']}
- %: {ytd['pct']}
- Meses reportados: {", ".join(ytd['meses']) if ytd['meses'] else "N/A"}

KPIs Full Year (referencia):
- Actual FY: {fy['actual']}
- Budget FY: {fy['budget']}
- Var FY: {fy['var']}
- % FY: {fy['pct']}

Resumen mensual (tabla):
{resumen_mes.to_string(index=False)}

Forecast:
{("Conservador=" + str(fc['conservador']) + ", Tendencial=" + str(fc['tendencial']) + ", Optimista=" + str(fc['optimista'])) if fc else "N/A"}

Riesgo:
Nivel={riesgo['level']} Score={riesgo['score']}/6 Meses_sin_datos={riesgo['missing_months']} Volatilidad_std={riesgo['volatilidad']}

Entrega en este formato:
1) Resumen ejecutivo (máx 6 bullets)
2) Diagnóstico (qué explica el gap)
3) Recomendaciones (acciones concretas, priorizadas)
4) Riesgos y supuestos críticos
5) Próximos pasos (qué medir / qué decisión tomar)
"""
            st.session_state["analisis_ia"] = gemini_generate(prompt)

        if st.session_state["analisis_ia"]:
            st.markdown(st.session_state["analisis_ia"])

        st.markdown("### 📄 Exportar informe ejecutivo (PDF)")
        titulo = st.text_input("Título del informe", value="Informe Ejecutivo — Dashboard + IA Generativa (MBA)")
        if st.button("Generar PDF"):
            pdf_file = "informe_ejecutivo_ai.pdf"
            make_executive_pdf(
                file_path=pdf_file,
                titulo=titulo,
                kpi_ytd=ytd,
                kpi_fy=fy,
                forecast=fc,
                riesgo=riesgo,
                analisis_ia=st.session_state.get("analisis_ia", ""),
                tabla_mensual=resumen_mes
            )
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="⬇️ Descargar PDF",
                    data=f,
                    file_name=pdf_file,
                    mime="application/pdf"
                )

# -------- TAB 4 --------
with tab4:
    st.subheader("IA Técnica Preventa (Gemini + RAG Manual)")
    pdf_path = find_manual_pdf()

    if not pdf_path:
        st.warning("No encuentro el PDF del manual. Súbelo como 'Manual_tecnico_preventa.pdf' en la raíz o dentro de 'manual_tecnico/'.")
        st.stop()

    st.caption(f"Manual detectado: `{pdf_path}` | Modelo Gemini: {pick_gemini_model()}")

    manual_text = load_manual_text(pdf_path)
    if not manual_text:
        st.warning("Pude abrir el PDF, pero no pude extraer texto (posible PDF escaneado). Exporta desde Google Docs como PDF con texto.")
        st.stop()

    chunks = chunk_text(manual_text)

    if "chat_tecnico" not in st.session_state:
        st.session_state["chat_tecnico"] = []

    for m in st.session_state["chat_tecnico"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    pregunta = st.chat_input("Ej: Snack 250g, VFFS, vida útil 6 meses, cliente quiere bajar micras. ¿Qué ofrezco?")

    if pregunta:
        st.session_state["chat_tecnico"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        ctx = retrieve_chunks(chunks, pregunta, top_k=4)
        contexto_txt = "\n\n---\n\n".join([f"[Fragmento {i+1}]\n{c}" for i, c in enumerate(ctx)])

        prompt = f"""
Eres un Asistente Técnico de PREVENTA para empaque plástico flexible (bolsa/bobina).

REGLAS:
- Responde SOLO usando la evidencia del CONTEXTO del manual.
- Si el contexto NO contiene información suficiente, NO inventes: pide datos faltantes (producto, peso, vida útil, máquina VFFS/HFFS, barrera OTR/WVTR si aplica, calibre, tipo de sello, ancho, tipo de material, uso final, condiciones de almacenamiento).
- Da dos opciones si aplica:
  A) segura (bajo riesgo de reclamo)
  B) optimizada costo (condicionada a prueba piloto)
- Menciona claramente riesgos técnicos y comerciales (reclamo, shelf-life, barrera, sellado).

FORMATO:
1) Recomendación A (segura)
2) Recomendación B (optimizada costo) — si aplica
3) Datos faltantes / supuestos
4) Riesgos (técnicos + comerciales)
5) Evidencia del manual (referencia a Fragmento 1/2/3…)

CONTEXTO (manual):
{contexto_txt if contexto_txt else "SIN CONTEXTO RELEVANTE ENCONTRADO."}

CONSULTA:
{pregunta}
"""
        respuesta = gemini_generate(prompt)

        with st.chat_message("assistant"):
            st.markdown(respuesta)

        st.session_state["chat_tecnico"].append({"role": "assistant", "content": respuesta})
