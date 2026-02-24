import os, re
import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="Dashboard + IA Generativa", layout="wide")

MANUAL_PATH = os.path.join("manual_tecnico", "Manual_tecnico_preventa.pdf")

# =========================
# FUNCIONES IA (Gemini)
# =========================

def gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.error("Falta GEMINI_API_KEY en Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

def gemini_generate(prompt):
    client = gemini_client()
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return resp.text

# =========================
# RAG SIMPLE PARA MANUAL
# =========================

@st.cache_data
def load_manual():
    reader = PdfReader(MANUAL_PATH)
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9áéíóúñ ]", " ", text.lower())
    return set(text.split())

def retrieve_chunks(manual_text, query):
    chunks = manual_text.split("\n\n")
    q = tokenize(query)
    scored = []
    for c in chunks:
        score = len(q.intersection(tokenize(c)))
        if score > 0:
            scored.append((score, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:4]]

# =========================
# DASHBOARD HELPERS
# =========================

def kpis(df):
    actual = df["actual_kg"].sum()
    budget = df["budget_kg"].sum()
    var = actual - budget
    pct = (actual/budget*100) if budget > 0 else 0
    return actual, budget, var, pct

# =========================
# UI
# =========================

st.title("📊 Dashboard + IA Generativa")

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Cargar Excel",
    "2) Dashboard",
    "3) IA del Dashboard",
    "4) IA Técnica Preventa"
])

# ====================================
# TAB 1 — CARGA
# ====================================

with tab1:
    ventas_file = st.file_uploader("Reporte de Ventas (.xlsx)", type=["xlsx"])
    pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"])

    if st.button("Procesar"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos.")
        else:
            dfv = pd.read_excel(ventas_file)
            dfp = pd.read_excel(pres_file)

            dfv["actual_kg"] = pd.to_numeric(dfv["KG"], errors="coerce").fillna(0)
            dfp["budget_kg"] = pd.to_numeric(dfp["KG"], errors="coerce").fillna(0)

            df = dfv.merge(dfp, on="ItemCode", how="left")
            df["budget_kg"] = df["budget_kg"].fillna(0)

            st.session_state["df"] = df
            st.success("Datos cargados.")

# ====================================
# TAB 2 — DASHBOARD
# ====================================

with tab2:
    if "df" not in st.session_state:
        st.warning("Carga archivos primero.")
    else:
        df = st.session_state["df"]
        actual, budget, var, pct = kpis(df)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Actual KG", f"{actual:,.0f}")
        c2.metric("Budget KG", f"{budget:,.0f}")
        c3.metric("Varianza", f"{var:,.0f}")
        c4.metric("% Cumplimiento", f"{pct:.1f}%")

# ====================================
# TAB 3 — IA DEL DASHBOARD
# ====================================

with tab3:
    if "df" not in st.session_state:
        st.warning("Carga datos primero.")
    else:
        df = st.session_state["df"]
        actual, budget, var, pct = kpis(df)

        if st.button("Generar Análisis Ejecutivo con IA"):
            prompt = f"""
Eres analista financiero industrial.
Usa SOLO los siguientes datos:

Actual KG: {actual}
Budget KG: {budget}
Varianza: {var}
Cumplimiento %: {pct}

Entrega:
1) Resumen ejecutivo
2) Conclusiones
3) Recomendaciones comerciales
4) Riesgos
No inventes datos.
"""
            response = gemini_generate(prompt)
            st.markdown(response)

# ====================================
# TAB 4 — IA TÉCNICA PREVENTA
# ====================================

with tab4:
    if not os.path.exists(MANUAL_PATH):
        st.error("Manual no encontrado en repo.")
        st.stop()

    manual = load_manual()

    pregunta = st.chat_input("Ej: Snack 250g, VFFS, vida útil 6 meses.")

    if pregunta:
        with st.chat_message("user"):
            st.markdown(pregunta)

        context = retrieve_chunks(manual, pregunta)

        prompt = f"""
Eres ingeniero preventa en empaque flexible.

Responde SOLO usando el CONTEXTO del manual.
Si falta información, pide datos faltantes.
Entrega:
1) Opción A segura
2) Opción B optimizada costo
3) Riesgos
4) Evidencia usada

CONTEXTO:
{context}

PREGUNTA:
{pregunta}
"""
        respuesta = gemini_generate(prompt)

        with st.chat_message("assistant"):
            st.markdown(respuesta)
