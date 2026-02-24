# app.py
import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ===================== CONFIG =====================
st.set_page_config(page_title="Dashboard Ventas vs Presupuesto (KG)", layout="wide")

MESES_ORDEN = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

# Ventas (tu reporte): columnas de KG por mes
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

# Presupuesto: columnas por mes en KG (ENE..DIC)
MESES_PRES = {
    "Enero": "ENE", "Febrero": "FEB", "Marzo": "MAR", "Abril": "ABR",
    "Mayo": "MAY", "Junio": "JUN", "Julio": "JUL", "Agosto": "AGO",
    "Septiembre": "SEP", "Octubre": "OCT", "Noviembre": "NOV", "Diciembre": "DIC",
}

# ===================== HELPERS =====================
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def normalizar_ventas(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """
    Devuelve 1 fila por (anio, mes, cliente, sku) con actual_kg.
    Requiere en ventas:
      - SlpName (opcional para filtro si hubiera +1 vendedor)
      - Código de cliente/proveedor
      - Nombre de cliente/proveedor  (nuevo en tu Excel)
      - ItemCode
      - ItemName
      - columnas Ene_KG..Dic_KG
    """
    id_cols = [
        "SlpName",
        "Código de cliente/proveedor",
        "Nombre de cliente/proveedor",
        "ItemCode",
        "ItemName",
    ]
    faltan = [c for c in id_cols if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en Ventas: {faltan}")

    df = df.copy()
    df["Código de cliente/proveedor"] = _strip_series(df["Código de cliente/proveedor"])
    df["Nombre de cliente/proveedor"] = _strip_series(df["Nombre de cliente/proveedor"])
    df["ItemCode"] = _strip_series(df["ItemCode"])

    out = []
    for mes, col_kg in MESES_VENTAS.items():
        if col_kg not in df.columns:
            raise ValueError(f"Falta la columna '{col_kg}' en Ventas.")
        tmp = df[id_cols].copy()
        tmp["anio"] = anio
        tmp["mes"] = mes
        tmp["actual_kg"] = _num(df[col_kg])
        out.append(tmp)

    long = pd.concat(out, ignore_index=True)
    long["mes"] = pd.Categorical(long["mes"], categories=MESES_ORDEN, ordered=True)
    # Renombramos para calzar con presupuesto
    long = long.rename(columns={"Nombre de cliente/proveedor": "Nombre de cliente"})
    return long

def normalizar_presupuesto(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """
    Devuelve 1 fila por (anio, mes, cliente, sku) con budget_kg.
    Requiere en presupuesto:
      - Nombre de cliente
      - Clasificación (si existe; si no existe, se omite sin fallar)
      - ItemCode
      - Nombre SKU (si existe; si no existe, se omite sin fallar)
      - PAÍS (si existe; si no existe, se omite sin fallar)
      - columnas ENE..DIC
    """
    # Columnas mínimas (las demás son opcionales)
    required = ["Nombre de cliente", "ItemCode"]
    faltan = [c for c in required if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas mínimas en Presupuesto: {faltan}")

    df = df.copy()
    df["Nombre de cliente"] = _strip_series(df["Nombre de cliente"])
    df["ItemCode"] = _strip_series(df["ItemCode"])

    # Opcionales
    opt_cols = []
    for c in ["Clasificación", "Nombre SKU", "PAÍS"]:
        if c in df.columns:
            opt_cols.append(c)

    id_cols = ["Nombre de cliente", "ItemCode"] + opt_cols

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
    """
    Merge correcto por cliente + sku + mes (gap real por cliente).
    """
    merged = ventas_long.merge(
        pres_long,
        on=["anio", "mes", "ItemCode", "Nombre de cliente"],
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

def ultimo_mes_con_ventas(df_filtrado: pd.DataFrame) -> str | None:
    by_mes = (
        df_filtrado.groupby("mes", as_index=False)["actual_kg"]
        .sum()
        .sort_values("mes")
    )
    by_mes = by_mes[by_mes["actual_kg"] > 0]
    if by_mes.empty:
        return None
    return str(by_mes.iloc[-1]["mes"])

def pareto_gap_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pareto del GAP negativo por cliente (solo déficit).
    Retorna tabla con acumulado y % acumulado del déficit.
    """
    by_cliente = (
        df.groupby("Nombre de cliente", as_index=False)[["actual_kg", "budget_kg"]]
        .sum()
    )
    by_cliente["var_kg"] = by_cliente["actual_kg"] - by_cliente["budget_kg"]
    deficit = by_cliente[by_cliente["var_kg"] < 0].copy()
    if deficit.empty:
        return deficit
    deficit["deficit_kg"] = -deficit["var_kg"]
    deficit = deficit.sort_values("deficit_kg", ascending=False)
    total_def = deficit["deficit_kg"].sum()
    deficit["deficit_acum_kg"] = deficit["deficit_kg"].cumsum()
    deficit["deficit_acum_pct"] = (deficit["deficit_acum_kg"] / total_def) * 100 if total_def > 0 else 0
    return deficit

# ===================== UI =====================
st.title("📊 Dashboard Gerencial — Cumplimiento vs Presupuesto (KG)")
tab1, tab2, tab3 = st.tabs(["1) Cargar Excel", "2) Dashboard (KG)", "3) Asistente IA Técnico"])

# --------------------- TAB 1 ---------------------
with tab1:
    st.subheader("Carga mensual de data (Excel)")
    colA, colB = st.columns(2)

    with colA:
        anio_ventas = st.number_input("Año Ventas", min_value=2020, max_value=2035, value=2026, step=1)
        ventas_file = st.file_uploader("Sube tu 'Reporte de ventas' (.xlsx)", type=["xlsx"], key="ventas")

    with colB:
        anio_pres = st.number_input("Año Presupuesto", min_value=2020, max_value=2035, value=2026, step=1)
        pres_file = st.file_uploader("Sube tu 'Presupuesto de ventas' (.xlsx)", type=["xlsx"], key="pres")

    st.caption("Tip: si tus Excel tienen varias hojas, procura que la primera hoja sea la tabla principal.")

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

                st.session_state["ventas_long"] = ventas_long
                st.session_state["pres_long"] = pres_long
                st.session_state["df_final"] = df_final

                st.success("✅ Archivos procesados. Ve a la pestaña 'Dashboard (KG)'.")
                with st.expander("Ver muestra de datos procesados"):
                    st.write("Ventas normalizadas (primeras 10 filas):")
                    st.dataframe(ventas_long.head(10), use_container_width=True)
                    st.write("Presupuesto normalizado (primeras 10 filas):")
                    st.dataframe(pres_long.head(10), use_container_width=True)

            except Exception as e:
                st.exception(e)

# --------------------- TAB 2 ---------------------
with tab2:
    st.subheader("Cumplimiento vs Presupuesto (KG)")

    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df_all = st.session_state["df_final"].copy()

        # ========== SIDEBAR FILTROS ==========
        st.sidebar.header("Filtros")
        anio = st.sidebar.selectbox("Año", sorted(df_all["anio"].unique()))
        df_all = df_all[df_all["anio"] == anio]

        # Si mañana hay más vendedores, habilitamos filtro; si solo hay 1 (Diana), no mostramos nada.
        if "SlpName" in df_all.columns:
            vendedores = sorted(df_all["SlpName"].dropna().unique())
            if len(vendedores) > 1:
                vend_sel = st.sidebar.multiselect("Vendedor", vendedores, default=vendedores)
                if vend_sel:
                    df_all = df_all[df_all["SlpName"].isin(vend_sel)]

        mes_sel = st.sidebar.multiselect("Mes (para análisis)", MESES_ORDEN, default=MESES_ORDEN)

        # ========== BLOQUE EJECUTIVO (YTD automático + Proyección) ==========
        st.markdown("## 🧭 Ejecutivo (YTD automático + Proyección)")

        ultimo_mes = ultimo_mes_con_ventas(df_all)

        if ultimo_mes is None:
            st.warning("No hay ventas (KG) para el año/filtros seleccionados. No se puede calcular YTD automático.")
        else:
            idx_ultimo = MESES_ORDEN.index(ultimo_mes)
            meses_ytd = MESES_ORDEN[: idx_ultimo + 1]
            df_ytd = df_all[df_all["mes"].isin(meses_ytd)].copy()

            actual_ytd = float(df_ytd["actual_kg"].sum())
            budget_ytd = float(df_ytd["budget_kg"].sum())
            var_ytd = actual_ytd - budget_ytd
            cumpl_ytd = (actual_ytd / budget_ytd * 100) if budget_ytd > 0 else 0.0

            budget_anual = float(df_all["budget_kg"].sum())

            meses_transcurridos = len(meses_ytd)
            run_rate = (actual_ytd / meses_transcurridos) if meses_transcurridos > 0 else 0.0
            proyeccion_anual = run_rate * 12

            meses_restantes = 12 - meses_transcurridos
            meta_restante = budget_anual - actual_ytd
            kg_necesarios_mes = (meta_restante / meses_restantes) if meses_restantes > 0 else 0.0

            proy_pct = (proyeccion_anual / budget_anual * 100) if budget_anual > 0 else 0.0
            if proy_pct >= 100:
                semaforo = "🟢 Verde"
            elif proy_pct >= 95:
                semaforo = "🟡 Amarillo"
            else:
                semaforo = "🔴 Rojo"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Actual YTD (KG) | hasta {ultimo_mes}", f"{actual_ytd:,.0f}")
            c2.metric("Budget YTD (KG)", f"{budget_ytd:,.0f}")
            c3.metric("Varianza YTD (KG)", f"{var_ytd:,.0f}")
            c4.metric("% Cumplimiento YTD", f"{cumpl_ytd:.1f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Run Rate (KG/mes)", f"{run_rate:,.0f}")
            c6.metric("Proyección anual (KG)", f"{proyeccion_anual:,.0f}")
            c7.metric("Proyección vs Presupuesto anual", f"{proy_pct:.1f}%")
            c8.metric("Riesgo de cierre", semaforo)

            st.caption(
                f"KG necesarios por mes para cumplir la meta anual: {kg_necesarios_mes:,.0f} "
                f"(con {meses_restantes} meses restantes)"
            )

        st.divider()

        # ========== SECCIÓN ANÁLISIS (respeta filtro de meses) ==========
        st.markdown("## 🔎 Análisis (según meses seleccionados)")

        df = df_all.copy()
        if mes_sel:
            df = df[df["mes"].isin(mes_sel)]

        # KPIs de análisis
        m = kpis(df)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Actual (KG)", f"{m['actual']:,.0f}")
        a2.metric("Budget (KG)", f"{m['budget']:,.0f}")
        a3.metric("Varianza (KG)", f"{m['var']:,.0f}")
        a4.metric("% Cumplimiento", f"{m['cumpl']:.1f}%")

        # Tendencia mensual
        st.markdown("### Tendencia mensual (Actual vs Budget)")
        by_mes = (
            df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]]
            .sum()
            .sort_values("mes")
        )
        st.plotly_chart(px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True), use_container_width=True)

        # Top SKUs con mayor gap (negativo primero)
        st.markdown("### Top SKUs con mayor gap (KG)")
        by_sku = (
            df.groupby(["ItemCode", "ItemName"], as_index=False)[["actual_kg", "budget_kg"]]
            .sum()
        )
        by_sku["var_kg"] = by_sku["actual_kg"] - by_sku["budget_kg"]
        by_sku = by_sku.sort_values("var_kg", ascending=True)
        st.dataframe(by_sku.head(30), use_container_width=True)

        # GAP por cliente (negativo primero)
        st.markdown("### 📉 GAP por Cliente (KG)")
        by_cliente = (
            df.groupby("Nombre de cliente", as_index=False)[["actual_kg", "budget_kg"]]
            .sum()
        )
        by_cliente["var_kg"] = by_cliente["actual_kg"] - by_cliente["budget_kg"]
        by_cliente["cumpl_pct"] = by_cliente.apply(
            lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0,
            axis=1
        )
        by_cliente = by_cliente.sort_values("var_kg", ascending=True)
        st.dataframe(by_cliente.head(30), use_container_width=True)

        # Cumplimiento por Clasificación (si existe en presupuesto)
        if "Clasificación" in df.columns:
            st.markdown("### Cumplimiento por Clasificación (KG)")
            by_clas = (
                df.groupby("Clasificación", as_index=False)[["actual_kg", "budget_kg"]]
                .sum()
            )
            by_clas["cumpl_pct"] = by_clas.apply(
                lambda r: (r["actual_kg"] / r["budget_kg"] * 100) if r["budget_kg"] > 0 else 0,
                axis=1
            )
            by_clas = by_clas.sort_values("cumpl_pct", ascending=False)
            st.plotly_chart(px.bar(by_clas, x="Clasificación", y="cumpl_pct"), use_container_width=True)

        # Pareto 80/20 del déficit (solo si hay gap negativo)
        st.markdown("### 🧩 Pareto del déficit (clientes que explican el GAP negativo)")
        pareto = pareto_gap_clientes(df)
        if pareto.empty:
            st.info("No hay déficit (gap negativo) en el período/filtros seleccionados.")
        else:
            st.dataframe(
                pareto[["Nombre de cliente", "deficit_kg", "deficit_acum_kg", "deficit_acum_pct"]].head(30),
                use_container_width=True
            )
            # Opcional: línea acumulada del % déficit
            chart = pareto.copy()
            chart["rank"] = range(1, len(chart) + 1)
            st.plotly_chart(px.line(chart, x="rank", y="deficit_acum_pct", markers=True), use_container_width=True)

        # Control: ventas sin presupuesto y presupuesto sin ventas
        st.markdown("### 🧪 Controles (calidad del cruce)")
        ccol1, ccol2 = st.columns(2)

        with ccol1:
            st.markdown("**Ventas sin presupuesto (budget = 0 y actual > 0)**")
            sin_pres = df[(df["budget_kg"] == 0) & (df["actual_kg"] > 0)].copy()
            show_cols = ["Nombre de cliente", "ItemCode", "ItemName", "mes", "actual_kg", "budget_kg"]
            show_cols = [c for c in show_cols if c in sin_pres.columns]
            st.dataframe(
                sin_pres[show_cols].sort_values("actual_kg", ascending=False).head(50),
                use_container_width=True
            )

        with ccol2:
            st.markdown("**Presupuesto sin ventas (actual = 0 y budget > 0)**")
            sin_ventas = df[(df["actual_kg"] == 0) & (df["budget_kg"] > 0)].copy()
            show_cols2 = ["Nombre de cliente", "ItemCode", "mes", "actual_kg", "budget_kg"]
            show_cols2 = [c for c in show_cols2 if c in sin_ventas.columns]
            st.dataframe(
                sin_ventas[show_cols2].sort_values("budget_kg", ascending=False).head(50),
                use_container_width=True
            )

# --------------------- TAB 3 ---------------------
with tab3:
    st.subheader("Asistente IA técnico (basado en tu manual)")
    st.caption("Responderá basado en tu manual. Si no hay evidencia, pedirá datos faltantes (no inventa).")

    st.info(
        "Para activarlo necesitas agregar OPENAI_API_KEY y OPENAI_VECTOR_STORE_ID en Streamlit Secrets. "
        "Si no están, esta sección no falla: solo queda desactivada."
    )

    # Chat UI
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Pregunta técnica (ej: bobina para detergente, ¿qué estructura y calibre sugieres?)")

    if user_q:
        st.session_state["chat"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            api_key = None
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", None)
            except Exception:
                api_key = os.environ.get("OPENAI_API_KEY")

            if not api_key:
                st.markdown("🔒 Asistente desactivado: configura `OPENAI_API_KEY` en Secrets.")
            else:
                vector_store_id = None
                try:
                    vector_store_id = st.secrets.get("OPENAI_VECTOR_STORE_ID", None)
                except Exception:
                    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")

                if not vector_store_id:
                    st.markdown("🔒 Falta `OPENAI_VECTOR_STORE_ID` en Secrets para usar tu manual con File Search.")
                else:
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)

                        system_instructions = """
Eres un asistente técnico para empaque plástico flexible.
Reglas:
- Responde SOLO usando información encontrada en el manual (file search).
- Si no hay evidencia suficiente, NO inventes: pide datos faltantes (producto, máquina, ancho, calibre, barrera OTR/WVTR, tipo de sello, etc.).
- Da respuesta en formato:
  1) Recomendación
  2) Datos faltantes / Supuestos
  3) Fuente (qué parte del manual respalda)
"""

                        resp = client.responses.create(
                            model="gpt-4.1-mini",
                            input=[
                                {"role": "system", "content": system_instructions},
                                {"role": "user", "content": user_q},
                            ],
                            tools=[{
                                "type": "file_search",
                                "vector_store_ids": [vector_store_id]
                            }],
                        )

                        answer = resp.output_text
                        st.markdown(answer)
                        st.session_state["chat"].append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error("Error llamando a OpenAI. Revisa API key, vector store y permisos.")
                        st.exception(e)
