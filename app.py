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

@st.cache_data(show_spinner=False)
def pick_gemini_model():
    """
    Elige un modelo disponible que soporte generateContent.
    Prioriza modelos Flash / Preview recientes.
    """
    client = gemini_client()
    models = client.models.list()

    # Candidatos recomendados (más nuevos primero)
    preferred = [
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    available = []
    for m in models:
        name = getattr(m, "name", "")
        if not name:
            continue
        model_id = name.replace("models/", "")

        # Revisa acciones soportadas si el SDK lo trae
        actions = getattr(m, "supported_actions", None) or getattr(m, "supportedActions", None)
        if actions and ("generateContent" not in actions):
            continue

        available.append(model_id)

    # 1) por preferencia
    for p in preferred:
        if p in available:
            return p

    # 2) fallback: cualquier modelo que parezca texto y funcione
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

        resp = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return resp.text or "(Sin respuesta)"
    except Exception as e:
        return f"❌ Error llamando a Gemini: {type(e).__name__} — {e}"

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
        df = st.session_state["df"].copy()

        # 1) Resumen mensual
        by_mes = df.groupby("mes", as_index=False)[["actual_kg","budget_kg"]].sum()

        # orden correcto de meses
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        # 2) Meses "reportados": donde ya hay ventas cargadas (>0)
        by_mes["reportado"] = by_mes["actual_kg"] > 0
        meses_reportados = by_mes[by_mes["reportado"]]["mes"].tolist()

        # Si todavía no hay meses con ventas > 0, evita división
        if len(meses_reportados) == 0:
            st.warning("Aún no hay meses con ventas cargadas (actual_kg > 0).")
            st.dataframe(by_mes)
            st.stop()

        # 3) KPIs YTD (solo meses reportados)
        ytd = by_mes[by_mes["reportado"]].copy()
# Solo meses con datos reales cargados
df_activos = df[df["actual_kg"] > 0]

total_actual = df_activos["actual_kg"].sum()
total_budget = df_activos["budget_kg"].sum()
total_var = total_actual - total_budget
total_pct = (total_actual / total_budget * 100) if total_budget > 0 else 0
by_mes = df_activos.groupby("mes", as_index=False)[["actual_kg","budget_kg"]].sum()
        # 4) KPIs Full Year (opcional, referencia)
        fy_actual = float(by_mes["actual_kg"].sum())
        fy_budget = float(by_mes["budget_kg"].sum())
        fy_pct = (fy_actual / fy_budget * 100) if fy_budget > 0 else 0.0

        st.caption(f"Meses reportados (ventas cargadas): {', '.join(meses_reportados)}")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Actual (KG) — YTD", f"{total_actual:,.0f}")
        c2.metric("Budget (KG) — YTD", f"{total_budget:,.0f}")
        c3.metric("Varianza (KG) — YTD", f"{total_var:,.0f}")
        c4.metric("% Cumplimiento — YTD", f"{total_pct:.1f}%")

        with st.expander("Ver referencia anual (Full Year)"):
            st.write(f"Actual FY: {fy_actual:,.0f} KG")
            st.write(f"Budget FY: {fy_budget:,.0f} KG")
            st.write(f"% Cumplimiento FY: {fy_pct:.1f}%")

        # 5) Gráfica: solo meses reportados (para que no “aplane”)
        by_mes_ytd = by_mes[by_mes["reportado"]].copy()

        st.plotly_chart(
            px.line(by_mes_ytd, x="mes", y=["actual_kg","budget_kg"], markers=True),
            use_container_width=True
        )

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
