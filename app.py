import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Dashboard Ventas vs Presupuesto (KG)", layout="wide")

MESES_ORDEN = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

MESES_VENTAS = {
    "Enero": ("Enero_QTZ", "Ene_KG"),
    "Febrero": ("Febrero_QTZ", "Feb_KG"),
    "Marzo": ("Marzo_QTZ", "Mar_KG"),
    "Abril": ("Abril_QTZ", "Abr_KG"),
    "Mayo": ("Mayo_QTZ", "May_KG"),
    "Junio": ("Junio_QTZ", "Jun_KG"),
    "Julio": ("Julio_QTZ", "Jul_KG"),
    "Agosto": ("Agosto_QTZ", "Ago_KG"),
    "Septiembre": ("Septiembre_QTZ", "Sep_KG"),
    "Octubre": ("Octubre_QTZ", "Oct_KG"),
    "Noviembre": ("Noviembre_QTZ", "Nov_KG"),
    "Diciembre": ("Diciembre_QTZ", "Dic_KG"),
}

MESES_PRES = {
    "Enero": "ENE", "Febrero": "FEB", "Marzo": "MAR", "Abril": "ABR",
    "Mayo": "MAY", "Junio": "JUN", "Julio": "JUL", "Agosto": "AGO",
    "Septiembre": "SEP", "Octubre": "OCT", "Noviembre": "NOV", "Diciembre": "DIC",
}

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def normalizar_ventas(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    id_cols = ["SlpName", "Código de cliente/proveedor", "ItemCode", "ItemName"]
    faltan = [c for c in id_cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en Ventas: {faltan}")

    out = []
    for mes, (_col_qtz, col_kg) in MESES_VENTAS.items():
        if col_kg not in df.columns:
            raise ValueError(f"Falta la columna '{col_kg}' en Ventas.")
        tmp = df[id_cols].copy()
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["actual_kg"] = _num(df[col_kg])
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    return long

def normalizar_presupuesto(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    id_cols = ["Nombre de cliente", "Clasificación", "ItemCode", "Nombre SKU", "PAÍS"]
    faltan = [c for c in id_cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en Presupuesto: {faltan}")

    out = []
    for mes, col_kg in MESES_PRES.items():
        if col_kg not in df.columns:
            raise ValueError(f"Falta la columna '{col_kg}' en Presupuesto.")
        tmp = df[id_cols].copy()
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["budget_kg"] = _num(df[col_kg])
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    return long

def calcular_cumplimiento(ventas_long: pd.DataFrame, pres_long: pd.DataFrame) -> pd.DataFrame:
    merged = ventas_long.merge(
        pres_long,
        on=["anio", "mes", "ItemCode"],
        how="left",
        suffixes=("_act", "_bud")
    )
    merged["budget_kg"] = merged["budget_kg"].fillna(0)
    merged["var_kg"] = merged["actual_kg"] - merged["budget_kg"]
    merged["cumpl_pct"] = (merged["actual_kg"] / merged["budget_kg"]).replace([float("inf")], 0).fillna(0) * 100
    return merged

def kpis(df: pd.DataFrame) -> dict:
    actual = float(df["actual_kg"].sum())
    budget = float(df["budget_kg"].sum())
    var = actual - budget
    cumpl = (actual / budget * 100) if budget > 0 else 0.0
    return {"actual": actual, "budget": budget, "var": var, "cumpl": cumpl}

st.title("📊 Dashboard Gerencial — Cumplimiento vs Presupuesto (KG)")
tab1, tab2, tab3 = st.tabs(["1) Cargar Excel", "2) Dashboard (KG)", "3) Asistente IA Técnico"])

with tab1:
    st.subheader("Carga mensual de data (Excel)")
    colA, colB = st.columns(2)

    with colA:
        anio_ventas = st.number_input("Año Ventas", min_value=2020, max_value=2035, value=2026, step=1)
        ventas_file = st.file_uploader("Sube tu 'Reporte de ventas' (.xlsx)", type=["xlsx"], key="ventas")

    with colB:
        anio_pres = st.number_input("Año Presupuesto", min_value=2020, max_value=2035, value=2026, step=1)
        pres_file = st.file_uploader("Sube tu 'Presupuesto de ventas' (.xlsx)", type=["xlsx"], key="pres")

    if st.button("Procesar archivos"):
        if ventas_file is None or pres_file is None:
            st.error("Sube ambos archivos: Ventas y Presupuesto.")
        else:
            try:
                df_ventas = pd.read_excel(ventas_file)
                df_pres = pd.read_excel(pres_file)

                ventas_long = normalizar_ventas(df_ventas, int(anio_ventas))
                pres_long = normalizar_presupuesto(df_pres, int(anio_pres))
                df_final = calcular_cumplimiento(ventas_long, pres_long)

                st.session_state["df_final"] = df_final
                st.success("✅ Listo. Ve a la pestaña 'Dashboard (KG)'.")
            except Exception as e:
                st.exception(e)

with tab2:
    st.subheader("Cumplimiento vs Presupuesto (KG)")
    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df = st.session_state["df_final"].copy()

        st.sidebar.header("Filtros")
        anio = st.sidebar.selectbox("Año", sorted(df["anio"].unique()))
        df = df[df["anio"] == anio]

        mes_sel = st.sidebar.multiselect("Mes", MESES_ORDEN, default=MESES_ORDEN)
        if mes_sel:
            df = df[df["mes"].isin(mes_sel)]

        vendedores = sorted(df["SlpName"].dropna().unique())
        vend_sel = st.sidebar.multiselect("Vendedor", vendedores, default=vendedores)
        if vend_sel:
            df = df[df["SlpName"].isin(vend_sel)]

        m = kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual (KG)", f"{m['actual']:,.0f}")
        c2.metric("Budget (KG)", f"{m['budget']:,.0f}")
        c3.metric("Varianza (KG)", f"{m['var']:,.0f}")
        c4.metric("% Cumplimiento", f"{m['cumpl']:.1f}%")

        st.markdown("### Tendencia mensual (Actual vs Budget)")
        by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum().sort_values("mes")
        st.plotly_chart(px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True), use_container_width=True)

        st.markdown("### Cumplimiento por vendedor (%)")
        by_vend = df.groupby("SlpName", as_index=False)[["actual_kg","budget_kg"]].sum()
        by_vend["cumpl_pct"] = by_vend.apply(lambda r: (r["actual_kg"]/r["budget_kg"]*100) if r["budget_kg"]>0 else 0, axis=1)
        by_vend = by_vend.sort_values("cumpl_pct", ascending=False)
        st.plotly_chart(px.bar(by_vend, x="SlpName", y="cumpl_pct"), use_container_width=True)

with tab3:
    st.subheader("Asistente IA técnico (basado en tu manual)")
    st.info("Si aún no configuraste OPENAI_API_KEY, esta sección no llamará a IA (no debe fallar).")

    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        st.warning("Configura OPENAI_API_KEY en Streamlit Secrets para activar el asistente.")
    else:
        st.success("API Key detectada ✅ (falta configurar Vector Store si usarás File Search).")
        st.write("Aquí integraremos tu manual cuando lo tengas listo.")
