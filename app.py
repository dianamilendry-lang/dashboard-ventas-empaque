import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="Dashboard + IA Generativa", layout="wide")

# ===================== CONFIG =====================
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

MANUAL_PATH = "Manual_tecnico_preventa.pdf"

# ===================== GEMINI =====================
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

# ===================== MANUAL =====================
@st.cache_data
def load_manual_text():
    if not os.path.exists(MANUAL_PATH):
        return None
    reader = PdfReader(MANUAL_PATH)
    txt = []
    for p in reader.pages:
        txt.append(p.extract_text() or "")
    return "\n".join(txt)

# ===================== NORMALIZACIÓN =====================
def normalizar_ventas(df):
    out = []
    for mes, col in MESES_VENTAS.items():
        tmp = df[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["actual_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out.append(tmp)
    return pd.concat(out)

def normalizar_pres(df):
    out = []
    for mes, col in MESES_PRES.items():
        tmp = df[["ItemCode"]].copy()
        tmp["mes"] = mes
        tmp["budget_kg"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out.append(tmp)
    return pd.concat(out)

# ===================== UI =====================
st.title("📊 Dashboard Mensual + IA Generativa")

tab1, tab2, tab3, tab4 = st.tabs([
    "1) Cargar Excel",
    "2) Dashboard",
    "3) IA del Dashboard",
    "4) IA Técnica Preventa"
])

# -------- TAB 1 --------
with tab1:
    ventas_file = st.file_uploader("Reporte Ventas (.xlsx)", type=["xlsx"])
    pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"])

    if st.button("Procesar"):
        if ventas_file and pres_file:
            dfv = pd.read_excel(ventas_file)
            dfp = pd.read_excel(pres_file)

            ventas_long = normalizar_ventas(dfv)
            pres_long = normalizar_pres(dfp)

            df = ventas_long.merge(pres_long, on=["ItemCode","mes"], how="left")
            df["budget_kg"] = df["budget_kg"].fillna(0)
            df["var_kg"] = df["actual_kg"] - df["budget_kg"]
            df["cumpl_pct"] = (df["actual_kg"]/df["budget_kg"]).replace([float("inf")],0).fillna(0)*100

            st.session_state["df"] = df
            st.success("Datos procesados")
        else:
            st.error("Sube ambos archivos")

# -------- TAB 2 --------
with tab2:
    if "df" in st.session_state:
        df = st.session_state["df"]
        total_actual = df["actual_kg"].sum()
        total_budget = df["budget_kg"].sum()
        total_pct = (total_actual/total_budget*100) if total_budget>0 else 0

        c1,c2,c3 = st.columns(3)
        c1.metric("Actual KG", f"{total_actual:,.0f}")
        c2.metric("Budget KG", f"{total_budget:,.0f}")
        c3.metric("% Cumplimiento", f"{total_pct:.1f}%")

        by_mes = df.groupby("mes", as_index=False)[["actual_kg","budget_kg"]].sum()
        st.plotly_chart(px.line(by_mes, x="mes", y=["actual_kg","budget_kg"], markers=True), use_container_width=True)
    else:
        st.warning("Carga datos primero")

# -------- TAB 3 --------
with tab3:
    if "df" in st.session_state:
        df = st.session_state["df"]
        resumen = df.groupby("mes")[["actual_kg","budget_kg","var_kg"]].sum().to_string()

        if st.button("Generar análisis con IA"):
            prompt = f"""
Eres analista financiero industrial.
Usa SOLO estos datos:

{resumen}

Entrega:
1) Resumen ejecutivo
2) Conclusiones
3) Recomendaciones
No inventes cifras.
"""
            st.markdown(gemini_generate(prompt))
    else:
        st.warning("Carga datos primero")

# -------- TAB 4 --------
with tab4:
    manual = load_manual_text()
    if not manual:
        st.error("No se encontró el manual en el repo.")
        st.stop()

    pregunta = st.chat_input("Ej: Café 500g, VFFS, 12 meses.")

    if pregunta:
        prompt = f"""
Eres ingeniero preventa en empaque flexible.

Responde SOLO usando el CONTEXTO.
Si falta información, pide datos faltantes.

Entrega:
1) Opción A segura
2) Opción B optimizada costo
3) Riesgos
4) Evidencia usada

CONTEXTO:
{manual[:5000]}

PREGUNTA:
{pregunta}
"""
        respuesta = gemini_generate(prompt)

        with st.chat_message("assistant"):
            st.markdown(respuesta)
