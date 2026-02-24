import os
import re
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ===================== OPTIONAL IMPORTS (avoid crash) =====================
GENAI_AVAILABLE = True
try:
    from google import genai
except Exception:
    GENAI_AVAILABLE = False

PDF_AVAILABLE = True
try:
    from PyPDF2 import PdfReader
except Exception:
    PDF_AVAILABLE = False


# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Dashboard Mensual + IA Generativa", layout="wide")
st.title("📊 Dashboard Mensual + IA Generativa")


# ===================== CONSTANTS =====================
MESES_ORDEN = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]
MES_A_NUM = {m: i + 1 for i, m in enumerate(MESES_ORDEN)}
NUM_A_MES = {i + 1: m for i, m in enumerate(MESES_ORDEN)}

# Ventas (tu archivo de ventas usa columnas *_KG por mes)
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

# Presupuesto (tu archivo de presupuesto usa ENE/FEB/... etc)
MESES_PRES = {
    "Enero": "ENE", "Febrero": "FEB", "Marzo": "MAR", "Abril": "ABR",
    "Mayo": "MAY", "Junio": "JUN", "Julio": "JUL", "Agosto": "AGO",
    "Septiembre": "SEP", "Octubre": "OCT", "Noviembre": "NOV", "Diciembre": "DIC",
}

MANUAL_DIR = "manual_tecnico"
MANUAL_DEFAULT_ROOT = "Manual_tecnico_preventa.pdf"


# ===================== GEMINI HELPERS =====================
def get_gemini_key() -> str | None:
    # Streamlit Cloud: st.secrets["GEMINI_API_KEY"]
    # Local: export GEMINI_API_KEY=...
    return (st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None) or os.getenv("GEMINI_API_KEY")


def gemini_client():
    if not GENAI_AVAILABLE:
        raise RuntimeError("Falta instalar `google-genai` (agrega `google-genai` a requirements.txt).")
    api_key = get_gemini_key()
    if not api_key:
        raise RuntimeError("Falta `GEMINI_API_KEY` en Secrets (o variable de entorno).")
    return genai.Client(api_key=api_key)


def gemini_generate(prompt: str, model_name: str):
    """
    Llama Gemini con manejo de errores claro.
    Model recomendado (según docs recientes): gemini-3-flash-preview
    """
    client = gemini_client()
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return getattr(resp, "text", "") or ""


# ===================== MANUAL PDF HELPERS =====================
@st.cache_data(show_spinner=False)
def find_manual_pdf():
    """
    Busca el PDF en:
    1) ./manual_tecnico/*.pdf
    2) ./Manual_tecnico_preventa.pdf (raíz)
    3) cualquier *.pdf en raíz (fallback)
    """
    # 1) manual_tecnico/
    if os.path.isdir(MANUAL_DIR):
        pdfs = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
        pdfs.sort()
        if pdfs:
            return os.path.join(MANUAL_DIR, pdfs[0])

    # 2) raíz con nombre esperado
    if os.path.isfile(MANUAL_DEFAULT_ROOT):
        return MANUAL_DEFAULT_ROOT

    # 3) fallback: cualquier pdf en raíz
    root_pdfs = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    root_pdfs.sort()
    if root_pdfs:
        return root_pdfs[0]

    return None


@st.cache_data(show_spinner=False)
def load_manual_text(pdf_path: str) -> str:
    if not PDF_AVAILABLE:
        return ""
    try:
        reader = PdfReader(pdf_path)
        chunks = []
        for p in reader.pages:
            chunks.append(p.extract_text() or "")
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def chunk_text(text: str, max_chars: int = 1400):
    """
    Chunk simple por párrafos para RAG ligero sin dependencias extra.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def simple_retrieve(chunks, query: str, k: int = 5):
    """
    Recuperación simple por overlap de palabras (sin sklearn).
    """
    q = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", query.lower())
    qset = set(q)
    if not qset:
        return chunks[:k]

    scored = []
    for idx, ch in enumerate(chunks):
        words = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+", ch.lower())
        if not words:
            scored.append((0, idx))
            continue
        wset = set(words)
        overlap = len(qset.intersection(wset))
        # pequeño boost si aparece una frase literal
        boost = 2 if query.lower() in ch.lower() else 0
        scored.append((overlap + boost, idx))
    scored.sort(reverse=True)
    top = [chunks[i] for _, i in scored[:k]]
    return top


# ===================== DATA NORMALIZATION =====================
def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def normalizar_ventas(dfv: pd.DataFrame, anio: int) -> pd.DataFrame:
    """
    Genera long format: anio, mes, actual_kg (+ llaves si existen)
    """
    # llaves opcionales (si existen, las preservamos)
    possible_keys = [
        "SlpName",
        "Código de cliente/proveedor",
        "Codigo de cliente",
        "Código de cliente",
        "Nombre de cliente",
        "ItemCode",
        "ItemName",
        "UM",
    ]
    keys = [c for c in possible_keys if c in dfv.columns]

    # asegurar ItemCode (muy importante para merge)
    if "ItemCode" not in dfv.columns:
        raise ValueError("El reporte de ventas debe contener la columna 'ItemCode'.")

    out = []
    for mes, col_kg in MESES_VENTAS.items():
        if col_kg not in dfv.columns:
            # si falta la columna del mes, asumimos 0 (no crashea)
            serie = pd.Series([0] * len(dfv))
        else:
            serie = pd.to_numeric(dfv[col_kg], errors="coerce").fillna(0)

        tmp = dfv[keys].copy()
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["actual_kg"] = serie
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    return long


def normalizar_presupuesto(dfp: pd.DataFrame, anio: int) -> pd.DataFrame:
    """
    Genera long format: anio, mes, budget_kg (+ ItemCode como llave)
    """
    if "ItemCode" not in dfp.columns:
        raise ValueError("El presupuesto debe contener la columna 'ItemCode'.")

    keys = [c for c in ["ItemCode", "Nombre de cliente", "Nombre SKU", "Clasificación", "PAÍS"] if c in dfp.columns]
    out = []

    for mes, col in MESES_PRES.items():
        if col not in dfp.columns:
            serie = pd.Series([0] * len(dfp))
        else:
            serie = pd.to_numeric(dfp[col], errors="coerce").fillna(0)

        tmp = dfp[keys].copy()
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["budget_kg"] = serie
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    return long


def merge_actual_budget(ventas_long: pd.DataFrame, budget_long: pd.DataFrame) -> pd.DataFrame:
    """
    Merge por anio + mes + ItemCode
    """
    merged = ventas_long.merge(
        budget_long[["anio", "mes", "ItemCode", "budget_kg"]],
        on=["anio", "mes", "ItemCode"],
        how="left"
    )
    merged["budget_kg"] = merged["budget_kg"].fillna(0)
    merged["var_kg"] = merged["actual_kg"] - merged["budget_kg"]
    merged["cumpl_pct"] = np.where(merged["budget_kg"] > 0, (merged["actual_kg"] / merged["budget_kg"]) * 100, 0.0)
    return merged


def months_with_real_sales(df: pd.DataFrame) -> list[str]:
    by_mes = df.groupby("mes", as_index=False)["actual_kg"].sum()
    by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
    by_mes = by_mes.sort_values("mes")
    real = by_mes[by_mes["actual_kg"] > 0]["mes"].astype(str).tolist()
    return real


def kpis_on_months(df: pd.DataFrame, meses_kpi: list[str]) -> dict:
    """
    KPIs calculados solo en meses_kpi (para no penalizar meses futuros sin ventas).
    """
    d = df[df["mes"].isin(meses_kpi)].copy()
    actual = float(d["actual_kg"].sum())
    budget = float(d["budget_kg"].sum())
    var = actual - budget
    cumpl = (actual / budget * 100) if budget > 0 else 0.0
    return {"actual": actual, "budget": budget, "var": var, "cumpl": cumpl}


# ===================== FORECAST + RISK =====================
def forecast_trend(by_mes: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast simple por tendencia lineal sobre meses con actual>0.
    by_mes requiere columnas: mes, actual_kg, budget_kg
    """
    dfm = by_mes.copy()
    dfm["mes_num"] = dfm["mes"].map(MES_A_NUM)

    train = dfm[dfm["actual_kg"] > 0].dropna(subset=["mes_num"])
    if len(train) < 2:
        dfm["forecast_kg"] = np.nan
        return dfm

    x = train["mes_num"].astype(float).values
    y = train["actual_kg"].astype(float).values

    # y = a*x + b
    a, b = np.polyfit(x, y, 1)

    dfm["forecast_kg"] = dfm["mes_num"].astype(float) * a + b
    # no permitir negativos
    dfm["forecast_kg"] = dfm["forecast_kg"].clip(lower=0)
    return dfm


def risk_summary(by_mes: pd.DataFrame, meses_reales: list[str]) -> dict:
    """
    Riesgo ejecutivo basado en:
    - Cumplimiento YTD (meses reales)
    - Run-rate requerido para cerrar el gap anual vs budget anual
    """
    # anual
    budget_anual = float(by_mes["budget_kg"].sum())
    actual_ytd = float(by_mes[by_mes["mes"].isin(meses_reales)]["actual_kg"].sum())
    budget_ytd = float(by_mes[by_mes["mes"].isin(meses_reales)]["budget_kg"].sum())

    gap_anual = budget_anual - actual_ytd
    meses_restantes = 12 - len(meses_reales)

    run_rate_req = (gap_anual / meses_restantes) if meses_restantes > 0 else 0.0
    cumpl_ytd = (actual_ytd / budget_ytd * 100) if budget_ytd > 0 else 0.0

    # banderas simples
    flags = []
    if len(meses_reales) <= 2:
        flags.append("Datos aún tempranos (1–2 meses). Interpretar con cautela.")
    if cumpl_ytd < 85 and len(meses_reales) >= 3:
        flags.append("Riesgo alto de incumplimiento YTD (<85%).")
    if run_rate_req > 0 and budget_anual > 0:
        # si el run-rate requerido supera 130% del promedio presupuestado mensual
        avg_budget = budget_anual / 12 if budget_anual > 0 else 0
        if avg_budget > 0 and run_rate_req > 1.3 * avg_budget:
            flags.append("Run-rate requerido muy alto para cerrar el año (>130% del presupuesto mensual promedio).")

    return {
        "budget_anual": budget_anual,
        "actual_ytd": actual_ytd,
        "budget_ytd": budget_ytd,
        "cumpl_ytd": cumpl_ytd,
        "gap_anual": gap_anual,
        "run_rate_req": run_rate_req,
        "meses_restantes": meses_restantes,
        "flags": flags
    }


# ===================== UI TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cargar archivos (mensual)",
    "Dashboard",
    "IA del Dashboard",
    "Forecast & Riesgo",
    "IA Técnica Preventa",
])

# Sidebar common filters if data exists
def sidebar_filters(df: pd.DataFrame):
    st.sidebar.header("Filtros")

    anios = sorted(df["anio"].dropna().unique().tolist())
    anio = st.sidebar.selectbox("Año", anios, index=0)

    # meses disponibles en el df
    meses_df = [m for m in MESES_ORDEN if m in set(df["mes"].unique())]
    mes_sel = st.sidebar.multiselect("Mes (análisis)", meses_df, default=meses_df)

    # vendedor si existe
    vend_sel = None
    if "SlpName" in df.columns:
        vends = sorted([v for v in df["SlpName"].dropna().unique().tolist() if str(v).strip() != ""])
        if vends:
            vend_sel = st.sidebar.multiselect("Vendedor", vends, default=vends)

    # cliente si existe
    cli_sel = None
    cli_col = None
    for candidate in ["Nombre de cliente", "Nombre de cliente/proveedor", "Nombre de cliente proveedor", "Nombre cliente"]:
        if candidate in df.columns:
            cli_col = candidate
            break
    if cli_col:
        clis = sorted([c for c in df[cli_col].dropna().unique().tolist() if str(c).strip() != ""])
        if clis:
            cli_sel = st.sidebar.multiselect("Cliente", clis, default=clis)

    return anio, mes_sel, vend_sel, (cli_col, cli_sel)


# ===================== TAB 1: LOAD =====================
with tab1:
    st.subheader("1) Cargar archivos (mensual)")
    c1, c2 = st.columns(2)
    with c1:
        ventas_file = st.file_uploader("Reporte de Ventas mensual (.xlsx)", type=["xlsx"], key="ventas_upl")
    with c2:
        pres_file = st.file_uploader("Presupuesto (.xlsx)", type=["xlsx"], key="pres_upl")

    anio = st.number_input("Año", min_value=2000, max_value=2100, value=2026, step=1)

    if st.button("Procesar", type="primary"):
        if not ventas_file or not pres_file:
            st.error("Sube ambos archivos: Ventas y Presupuesto.")
        else:
            try:
                dfv = pd.read_excel(ventas_file)
                dfp = pd.read_excel(pres_file)

                ventas_long = normalizar_ventas(dfv, int(anio))
                pres_long = normalizar_presupuesto(dfp, int(anio))

                df = merge_actual_budget(ventas_long, pres_long)

                # guardamos todo
                st.session_state["df"] = df

                # meses con ventas reales
                reales = months_with_real_sales(df)
                st.success("✅ Datos procesados correctamente.")
                st.info("Meses detectados con datos reales: " + (", ".join(reales) if reales else "Ninguno (todo 0)"))

                with st.expander("Ver muestra (primeras filas)"):
                    st.dataframe(df.head(30), use_container_width=True)

                st.caption("Tip: Si tu Excel mensual solo tiene 1–2 meses con números, este sistema calcula KPIs solo sobre esos meses (no penaliza meses futuros).")

            except Exception as e:
                st.exception(e)


# ===================== TAB 2: DASHBOARD =====================
with tab2:
    st.subheader("2) Dashboard")

    if "df" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df = st.session_state["df"].copy()

        # filtros
        anio_sel, mes_sel, vend_sel, (cli_col, cli_sel) = sidebar_filters(df)
        df = df[df["anio"] == anio_sel]

        if mes_sel:
            df = df[df["mes"].isin(mes_sel)]
        if vend_sel and "SlpName" in df.columns:
            df = df[df["SlpName"].isin(vend_sel)]
        if cli_col and cli_sel:
            df = df[df[cli_col].isin(cli_sel)]

        # meses reales (dentro de filtros)
        meses_reales = months_with_real_sales(df)

        # el usuario puede decidir si KPIs usan SOLO meses reales o todos los meses seleccionados
        st.sidebar.divider()
        kpi_mode = st.sidebar.radio(
            "KPIs basados en:",
            ["Solo meses con ventas reales", "Todos los meses seleccionados"],
            index=0
        )
        meses_kpi = meses_reales if kpi_mode == "Solo meses con ventas reales" else (mes_sel if mes_sel else MESES_ORDEN)

        if not meses_kpi:
            st.warning("No hay meses con ventas reales en el filtro actual.")
            meses_kpi = mes_sel if mes_sel else MESES_ORDEN

        m = kpis_on_months(df, meses_kpi)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual (KG)", f"{m['actual']:,.0f}")
        c2.metric("Budget (KG)", f"{m['budget']:,.0f}")
        c3.metric("Varianza (KG)", f"{m['var']:,.0f}")
        c4.metric("% Cumplimiento", f"{m['cumpl']:.1f}%")

        st.markdown("### Tendencia mensual (Actual vs Budget)")
        by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum()
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        # Para el gráfico, si estás en "solo meses reales", mostramos esos; si no, mostramos todos.
        chart_df = by_mes[by_mes["mes"].astype(str).isin(meses_kpi)] if meses_kpi else by_mes

        st.plotly_chart(
            px.line(chart_df, x="mes", y=["actual_kg", "budget_kg"], markers=True),
            use_container_width=True
        )

        # vendedor
        if "SlpName" in df.columns:
            st.markdown("### Cumplimiento por vendedor (%)")
            by_v = df.groupby("SlpName", as_index=False)[["actual_kg", "budget_kg"]].sum()
            by_v["cumpl_pct"] = np.where(by_v["budget_kg"] > 0, (by_v["actual_kg"] / by_v["budget_kg"]) * 100, 0.0)
            by_v = by_v.sort_values("cumpl_pct", ascending=False)

            st.plotly_chart(px.bar(by_v, x="SlpName", y="cumpl_pct"), use_container_width=True)

        st.caption("KPIs y tendencia se adaptan a carga mensual: si solo hay ventas en 1–2 meses, no se penalizan los meses futuros.")


# ===================== TAB 3: IA DASHBOARD =====================
with tab3:
    st.subheader("IA del Dashboard (análisis ejecutivo)")
    st.caption("Genera un informe (sin inventar cifras) con base en los datos agregados por mes.")

    if "df" not in st.session_state:
        st.warning("Carga datos primero en la pestaña 1.")
    else:
        df = st.session_state["df"].copy()

        # modelo
        default_model = "gemini-3-flash-preview"
        model_name = st.text_input("Modelo Gemini", value=default_model, help="Ej: gemini-3-flash-preview")

        # agregación por mes
        by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg", "var_kg"]].sum()
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")

        # meses reales globales (sin filtro) para que el informe no castigue meses futuros
        meses_reales = months_with_real_sales(df)
        by_mes_real = by_mes[by_mes["mes"].astype(str).isin(meses_reales)] if meses_reales else by_mes

        # resumen en texto
        resumen = by_mes_real.to_string(index=False)

        if st.button("Generar análisis con IA", type="primary"):
            if not GENAI_AVAILABLE:
                st.error("Falta la librería `google-genai`. Agrega `google-genai` en requirements.txt.")
            else:
                prompt = f"""
Eres un consultor ejecutivo (MBA) y analista comercial-industrial.

REGLAS:
- Usa SOLO los datos del bloque DATOS.
- No inventes cifras.
- Si detectas que solo hay 1–2 meses con ventas, indícalo como 'corte parcial mensual' (no como caída).

DATOS (agregado por mes, solo meses con ventas reales):
{resumen}

ENTREGA en español, con bullets y números:
1) Resumen ejecutivo (5 bullets)
2) Hallazgos clave (máximo 7)
3) Recomendaciones accionables (máximo 7)
4) Riesgos y supuestos
"""
                try:
                    ans = gemini_generate(prompt, model_name=model_name)
                    st.markdown(ans if ans else "No recibí texto del modelo.")
                    st.download_button(
                        "Descargar informe (TXT)",
                        data=ans.encode("utf-8"),
                        file_name="informe_ia_dashboard.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error("No se pudo generar el análisis. Revisa tu GEMINI_API_KEY y el modelo.")
                    st.exception(e)


# ===================== TAB 4: FORECAST & RISK =====================
with tab4:
    st.subheader("Forecast & Riesgo (nivel consultoría ejecutiva)")

    if "df" not in st.session_state:
        st.warning("Carga datos primero en la pestaña 1.")
    else:
        df = st.session_state["df"].copy()

        by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum()
        by_mes["mes"] = pd.Categorical(by_mes["mes"], categories=MESES_ORDEN, ordered=True)
        by_mes = by_mes.sort_values("mes")
        by_mes["mes"] = by_mes["mes"].astype(str)

        meses_reales = months_with_real_sales(df)
        risk = risk_summary(by_mes, meses_reales)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual YTD (KG)", f"{risk['actual_ytd']:,.0f}")
        c2.metric("Budget YTD (KG)", f"{risk['budget_ytd']:,.0f}")
        c3.metric("Cumplimiento YTD", f"{risk['cumpl_ytd']:.1f}%")
        c4.metric("Gap anual vs Budget", f"{risk['gap_anual']:,.0f}")

        st.markdown("### Riesgos y banderas")
        if risk["flags"]:
            for f in risk["flags"]:
                st.warning(f)
        else:
            st.success("Sin banderas críticas con la información actual.")

        st.markdown("### Run-rate requerido para cerrar el año")
        st.write(
            f"Meses con ventas reales: **{len(meses_reales)}** ({', '.join(meses_reales) if meses_reales else 'Ninguno'}).  "
            f"Meses restantes: **{risk['meses_restantes']}**.  "
            f"Run-rate requerido: **{risk['run_rate_req']:,.0f} KG/mes**."
        )

        st.markdown("### Forecast (tendencia simple)")
        fc = forecast_trend(by_mes.rename(columns={"mes": "mes"}))
        # para forecast graficable, dejamos mes string y mes_num
        fc["mes"] = pd.Categorical(fc["mes"], categories=MESES_ORDEN, ordered=True)
        fc = fc.sort_values("mes")
        fc["mes"] = fc["mes"].astype(str)

        st.dataframe(fc[["mes", "actual_kg", "budget_kg", "forecast_kg"]], use_container_width=True)

        plot_df = fc.melt(id_vars=["mes"], value_vars=["actual_kg", "budget_kg", "forecast_kg"], var_name="serie", value_name="kg")
        st.plotly_chart(px.line(plot_df, x="mes", y="kg", color="serie", markers=True), use_container_width=True)

        st.caption("Forecast simple por tendencia lineal. Si solo hay 1 mes con ventas, no se calcula.")


# ===================== TAB 5: IA TÉCNICA PREVENTA =====================
with tab5:
    st.subheader("IA Técnica Preventa (basada en manual PDF del repo)")
    st.caption("Responde SOLO con evidencia del manual. Si falta información, pide los datos faltantes (no inventa).")

    if not GENAI_AVAILABLE:
        st.error("Falta la librería `google-genai`. Agrega `google-genai` en requirements.txt para usar Gemini.")
        st.stop()

    pdf_path = find_manual_pdf()
    if not pdf_path:
        st.warning("No encuentro el manual PDF. Súbelo al repo como `Manual_tecnico_preventa.pdf` (raíz) o dentro de `manual_tecnico/`.")
        st.stop()

    st.info(f"Manual detectado: `{pdf_path}`")

    manual_text = load_manual_text(pdf_path)
    if not manual_text:
        st.warning("Pude ubicar el PDF, pero no pude extraer texto. Si es escaneado, expórtalo como PDF con texto seleccionable.")
        st.stop()

    # modelo
    default_model = "gemini-3-flash-preview"
    model_name = st.text_input("Modelo Gemini (Técnico)", value=default_model, key="model_tec")

    # RAG ligero
    chunks = chunk_text(manual_text, max_chars=1400)

    if "chat_hist" not in st.session_state:
        st.session_state["chat_hist"] = []

    # Mostrar historial
    for role, content in st.session_state["chat_hist"]:
        with st.chat_message(role):
            st.markdown(content)

    pregunta = st.chat_input("Ej: Café 500g, VFFS, vida útil 12 meses.")
    if pregunta:
        st.session_state["chat_hist"].append(("user", pregunta))
        with st.chat_message("user"):
            st.markdown(pregunta)

        top_ctx = simple_retrieve(chunks, pregunta, k=6)
        context = "\n\n---\n\n".join(top_ctx)

        prompt = f"""
Eres ingeniero(a) de preventa en Empaque Plástico Flexible (Bolsa y Bobina).

REGLAS ESTRICTAS:
- Responde SOLO usando el CONTEXTO (manual).
- Si falta información para recomendar estructura/espesor/barrera, primero pide los datos faltantes (lista corta).
- No inventes normas, micrajes, capas, ni aditivos si no aparecen en el CONTEXTO.
- La salida debe ser práctica para cotizar y diseñar.

FORMATO DE SALIDA:
A) Datos faltantes (si aplica) en bullets
B) Opción A segura (máxima protección): estructura sugerida + micraje + notas de proceso + por qué
C) Opción B optimizada costo: estructura sugerida + micraje + notas + por qué
D) Riesgos/alertas
E) Evidencia usada: cita textual corta (1–2 líneas) del CONTEXTO, o indica "No hay evidencia suficiente" y pide datos.

CONTEXTO:
{context}

PREGUNTA:
{pregunta}
"""
        try:
            respuesta = gemini_generate(prompt, model_name=model_name)
        except Exception as e:
            respuesta = "❌ No pude generar respuesta con Gemini. Revisa tu GEMINI_API_KEY y el nombre del modelo."
            st.exception(e)

        st.session_state["chat_hist"].append(("assistant", respuesta))
        with st.chat_message("assistant"):
            st.markdown(respuesta)

    with st.expander("Diagnóstico (para debug)"):
        st.write("Contenido raíz:", os.listdir("."))
        if os.path.isdir(MANUAL_DIR):
            st.write("Contenido manual_tecnico/:", os.listdir(MANUAL_DIR))
        st.write("Chunks del manual:", len(chunks))
