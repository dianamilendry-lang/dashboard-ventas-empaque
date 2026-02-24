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

def _pct(a: float, b: float) -> float:
    return (a / b * 100) if b and b != 0 else 0.0

def normalizar_ventas(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """
    1 fila por (anio, mes, cliente, sku) con actual_kg.
    Requiere en ventas:
      - SlpName
      - Código de cliente/proveedor
      - Nombre de cliente/proveedor
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
    1 fila por (anio, mes, cliente, sku) con budget_kg.
    Mínimo: Nombre de cliente + ItemCode + ENE..DIC
    Opcionales: Clasificación, Nombre SKU, PAÍS
    """
    required = ["Nombre de cliente", "ItemCode"]
    faltan = [c for c in required if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas mínimas en Presupuesto: {faltan}")

    df = df.copy()
    df["Nombre de cliente"] = _strip_series(df["Nombre de cliente"])
    df["ItemCode"] = _strip_series(df["ItemCode"])

    opt_cols = [c for c in ["Clasificación", "Nombre SKU", "PAÍS"] if c in df.columns]
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

def generar_conclusiones(
    df_ytd: pd.DataFrame,
    df_anual: pd.DataFrame,
    df_periodo: pd.DataFrame,
    ultimo_mes: str | None,
    top_n: int = 5
) -> dict:
    # Ejecutivo
    actual_ytd = float(df_ytd["actual_kg"].sum())
    budget_ytd = float(df_ytd["budget_kg"].sum())
    var_ytd = actual_ytd - budget_ytd
    cumpl_ytd = _pct(actual_ytd, budget_ytd)

    budget_anual = float(df_anual["budget_kg"].sum())
    meses_transcurridos = int(df_ytd["mes"].nunique()) if not df_ytd.empty else 0
    run_rate = (actual_ytd / meses_transcurridos) if meses_transcurridos > 0 else 0.0
    proyeccion_anual = run_rate * 12
    proy_pct = _pct(proyeccion_anual, budget_anual)

    meses_restantes = 12 - meses_transcurridos
    meta_restante = budget_anual - actual_ytd
    kg_necesarios_mes = (meta_restante / meses_restantes) if meses_restantes > 0 else 0.0

    if proy_pct >= 100:
        semaforo = "🟢 En ruta / sobre meta"
    elif proy_pct >= 95:
        semaforo = "🟡 Riesgo moderado"
    else:
        semaforo = "🔴 Riesgo alto"

    # Periodo (análisis)
    actual_p = float(df_periodo["actual_kg"].sum())
    budget_p = float(df_periodo["budget_kg"].sum())
    var_p = actual_p - budget_p
    cumpl_p = _pct(actual_p, budget_p)

    # Top déficit clientes
    by_cliente = df_periodo.groupby("Nombre de cliente", as_index=False)[["actual_kg","budget_kg"]].sum()
    by_cliente["var_kg"] = by_cliente["actual_kg"] - by_cliente["budget_kg"]
    clientes_deficit = by_cliente[by_cliente["var_kg"] < 0].sort_values("var_kg").head(top_n)

    # Top déficit SKUs
    by_sku = df_periodo.groupby(["ItemCode","ItemName"], as_index=False)[["actual_kg","budget_kg"]].sum()
    by_sku["var_kg"] = by_sku["actual_kg"] - by_sku["budget_kg"]
    skus_deficit = by_sku[by_sku["var_kg"] < 0].sort_values("var_kg").head(top_n)

    # Pareto déficit
    pareto = pareto_gap_clientes(df_periodo)
    pareto_msg = None
    if not pareto.empty:
        n80 = int((pareto["deficit_acum_pct"] <= 80).sum())
        n80 = min(n80 + 1, len(pareto))
        total_def = float(pareto["deficit_kg"].sum())
        pareto_msg = f"Enfoque Pareto: ~{n80} clientes explican ~80% del déficit (déficit total {total_def:,.0f} KG)."

    # Controles calidad
    ventas_sin_pres = df_periodo[(df_periodo["budget_kg"] == 0) & (df_periodo["actual_kg"] > 0)]
    pres_sin_ventas = df_periodo[(df_periodo["actual_kg"] == 0) & (df_periodo["budget_kg"] > 0)]
    kgs_ventas_sin_pres = float(ventas_sin_pres["actual_kg"].sum())
    kgs_pres_sin_ventas = float(pres_sin_ventas["budget_kg"].sum())

    # Conclusiones
    conclusiones = []
    if ultimo_mes:
        conclusiones.append(
            f"**YTD hasta {ultimo_mes}:** {actual_ytd:,.0f} KG vs {budget_ytd:,.0f} KG "
            f"({cumpl_ytd:.1f}%); varianza {var_ytd:,.0f} KG."
        )
    conclusiones.append(
        f"**Proyección anual (run rate):** {proyeccion_anual:,.0f} KG vs {budget_anual:,.0f} KG "
        f"({proy_pct:.1f}%). Estado: {semaforo}."
    )
    if meses_restantes > 0:
        conclusiones.append(
            f"**Ritmo requerido:** faltan {meta_restante:,.0f} KG en {meses_restantes} meses "
            f"(~{kg_necesarios_mes:,.0f} KG/mes)."
        )
    conclusiones.append(
        f"**Periodo seleccionado:** {actual_p:,.0f} KG vs {budget_p:,.0f} KG ({cumpl_p:.1f}%), "
        f"varianza {var_p:,.0f} KG."
    )

    # Recomendaciones sugeridas (no decisiones)
    recomendaciones = []
    if proy_pct < 95:
        recomendaciones.append("Activar **plan de recuperación**: priorizar cuentas con déficit y revisar pipeline/mix.")
    elif proy_pct < 100:
        recomendaciones.append("Seguimiento **semanal**: cerrar brechas en clientes/SKUs deficitarios para asegurar cierre ≥100%.")

    if not clientes_deficit.empty:
        top3 = ", ".join([str(x) for x in clientes_deficit["Nombre de cliente"].head(3).tolist()])
        recomendaciones.append(f"Priorizar gestión en **Top {min(top_n, len(clientes_deficit))} clientes con déficit** (ej.: {top3}).")
    else:
        recomendaciones.append("No se observan clientes con déficit en el periodo seleccionado (según data cargada).")

    if pareto_msg:
        recomendaciones.append(pareto_msg)

    if not skus_deficit.empty:
        recomendaciones.append("Revisar **SKUs con déficit**: disponibilidad/lead time, competitividad y condiciones comerciales.")

    if kgs_ventas_sin_pres > 0:
        recomendaciones.append(f"Revisar **ventas sin presupuesto** ({kgs_ventas_sin_pres:,.0f} KG): posible actualización de metas/presupuesto.")
    if kgs_pres_sin_ventas > 0:
        recomendaciones.append(f"Revisar **presupuesto sin ventas** ({kgs_pres_sin_ventas:,.0f} KG): identificar cuentas/SKUs sin tracción.")

    # Riesgos
    riesgos = []
    if kgs_ventas_sin_pres > 0 or kgs_pres_sin_ventas > 0:
        riesgos.append("Desalineación entre presupuesto y ventas (cliente/SKU). Puede sesgar el cumplimiento por cliente.")
    riesgos.append("El cruce requiere consistencia de nombres de cliente y ItemCode entre ambos archivos.")

    return {
        "conclusiones": conclusiones,
        "recomendaciones": recomendaciones,
        "riesgos": riesgos,
        "semaforo": semaforo,
    }

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

        # Si mañana hay más vendedores, habilitamos filtro; si solo hay 1, no mostramos nada.
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
            st.warning("No hay ventas (KG) en el año/filtros seleccionados. No se puede calcular YTD automático.")
            df_ytd = df_all.iloc[0:0].copy()
        else:
            idx_ultimo = MESES_ORDEN.index(ultimo_mes)
            meses_ytd = MESES_ORDEN[: idx_ultimo + 1]
            df_ytd = df_all[df_all["mes"].isin(meses_ytd)].copy()

            actual_ytd = float(df_ytd["actual_kg"].sum())
            budget_ytd = float(df_ytd["budget_kg"].sum())
            var_ytd = actual_ytd - budget_ytd
            cumpl_ytd = _pct(actual_ytd, budget_ytd)

            budget_anual = float(df_all["budget_kg"].sum())

            meses_transcurridos = len(meses_ytd)
            run_rate = (actual_ytd / meses_transcurridos) if meses_transcurridos > 0 else 0.0
            proyeccion_anual = run_rate * 12

            meses_restantes = 12 - meses_transcurridos
            meta_restante = budget_anual - actual_ytd
            kg_necesarios_mes = (meta_restante / meses_restantes) if meses_restantes > 0 else 0.0

            proy_pct = _pct(proyeccion_anual, budget_anual)
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
            c8.metric("Semáforo", semaforo)

            st.caption(
                f"KG necesarios por mes para cumplir la meta anual: {kg_necesarios_mes:,.0f} "
                f"(con {meses_restantes} meses restantes)"
            )

        # ========== SECCIÓN ANÁLISIS (respeta filtro de meses) ==========
        st.divider()
        st.markdown("## 🔎 Análisis (según meses seleccionados)")
        df = df_all.copy()
        if mes_sel:
            df = df[df["mes"].isin(mes_sel)]

        m = kpis(df)
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Actual (KG)", f"{m['actual']:,.0f}")
        a2.metric("Budget (KG)", f"{m['budget']:,.0f}")
        a3.metric("Varianza (KG)", f"{m['var']:,.0f}")
        a4.metric("% Cumplimiento", f"{m['cumpl']:.1f}%")

        st.markdown("### Tendencia mensual (Actual vs Budget)")
        by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum().sort_values("mes")
        st.plotly_chart(px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True), use_container_width=True)

        # ========== CONCLUSIONES Y RECOMENDACIONES (VISIBLE SIEMPRE) ==========
        st.divider()
        st.markdown("## 🧾 Conclusiones y recomendaciones")

        if ultimo_mes is None or df_ytd.empty:
            st.info("No se generaron conclusiones porque no hay ventas > 0 para calcular YTD automático en los filtros seleccionados.")
        else:
            insights = generar_conclusiones(
                df_ytd=df_ytd,
                df_anual=df_all,
                df_periodo=df,
                ultimo_mes=ultimo_mes,
                top_n=5
            )

            colL, colR = st.columns([2, 1])
            with colL:
                st.markdown("### Conclusiones")
                for x in insights["conclusiones"]:
                    st.markdown(f"- {x}")

                st.markdown("### Recomendaciones sugeridas")
                for x in insights["recomendaciones"]:
                    st.markdown(f"- {x}")

            with colR:
                st.markdown("### Riesgos / supuestos")
                for x in insights["riesgos"]:
                    st.markdown(f"- {x}")

            # ========== RESUMEN PARA COMITÉ (copiar/pegar) con Acción 1 y 2 ==========
            st.markdown("### 🧷 Resumen para comité (copiar/pegar)")

            actual_ytd = float(df_ytd["actual_kg"].sum())
            budget_ytd = float(df_ytd["budget_kg"].sum())
            cumpl_ytd = _pct(actual_ytd, budget_ytd)

            budget_anual = float(df_all["budget_kg"].sum())
            meses_transcurridos = int(df_ytd["mes"].nunique()) if not df_ytd.empty else 0
            run_rate = (actual_ytd / meses_transcurridos) if meses_transcurridos > 0 else 0.0
            proyeccion_anual = run_rate * 12
            proy_pct = _pct(proyeccion_anual, budget_anual)

            meses_restantes = 12 - meses_transcurridos
            meta_restante = budget_anual - actual_ytd
            kg_necesarios_mes = (meta_restante / meses_restantes) if meses_restantes > 0 else 0.0

            by_cliente_tmp = df.groupby("Nombre de cliente", as_index=False)[["actual_kg","budget_kg"]].sum()
            by_cliente_tmp["var_kg"] = by_cliente_tmp["actual_kg"] - by_cliente_tmp["budget_kg"]
            top_def_cli = by_cliente_tmp[by_cliente_tmp["var_kg"] < 0].sort_values("var_kg").head(3)

            if top_def_cli.empty:
                top_cli_txt = "Sin déficit por cliente en el periodo seleccionado."
            else:
                top_cli_txt = "; ".join([f"{r['Nombre de cliente']} ({r['var_kg']:,.0f} KG)" for _, r in top_def_cli.iterrows()])

            by_sku_tmp = df.groupby(["ItemCode","ItemName"], as_index=False)[["actual_kg","budget_kg"]].sum()
            by_sku_tmp["var_kg"] = by_sku_tmp["actual_kg"] - by_sku_tmp["budget_kg"]
            top_def_sku = by_sku_tmp[by_sku_tmp["var_kg"] < 0].sort_values("var_kg").head(3)

            if top_def_sku.empty:
                top_sku_txt = "Sin déficit por SKU en el periodo seleccionado."
            else:
                top_sku_txt = "; ".join([f"{r['ItemCode']} ({r['var_kg']:,.0f} KG)" for _, r in top_def_sku.iterrows()])

            sin_pres_kg = float(df[(df["budget_kg"] == 0) & (df["actual_kg"] > 0)]["actual_kg"].sum())
            sin_ventas_kg = float(df[(df["actual_kg"] == 0) & (df["budget_kg"] > 0)]["budget_kg"].sum())
            riesgo_txt = f"Control: ventas sin presupuesto {sin_pres_kg:,.0f} KG; presupuesto sin ventas {sin_ventas_kg:,.0f} KG."

            pareto = pareto_gap_clientes(df)
            if not pareto.empty:
                n80 = int((pareto["deficit_acum_pct"] <= 80).sum())
                n80 = min(n80 + 1, len(pareto))
                top_clientes = pareto.head(min(5, len(pareto)))["Nombre de cliente"].tolist()
                top_clientes_txt = ", ".join([str(x) for x in top_clientes])
                accion_1 = f"Acción 1 (Pareto): enfoque en ~{n80} clientes que explican ~80% del déficit. Top: {top_clientes_txt}."
            else:
                accion_1 = "Acción 1 (Pareto): no aplica (no hay déficit por cliente en el periodo seleccionado)."

            if meses_restantes > 0:
                accion_2 = f"Acción 2 (Recuperación): sostener ~{kg_necesarios_mes:,.0f} KG/mes durante {meses_restantes} meses para cerrar ≥100%."
            else:
                accion_2 = "Acción 2 (Recuperación): no aplica (año completo / sin meses restantes)."

            resumen = "\n".join([
                f"1) YTD hasta {ultimo_mes}: {actual_ytd:,.0f} KG vs {budget_ytd:,.0f} KG ({cumpl_ytd:.1f}%).",
                f"2) Proyección anual (run rate): {proyeccion_anual:,.0f} KG vs {budget_anual:,.0f} KG ({proy_pct:.1f}%).",
                f"3) Top clientes con déficit (periodo): {top_cli_txt}",
                f"4) Top SKUs con déficit (periodo): {top_sku_txt}",
                f"5) {riesgo_txt}",
                f"6) {accion_1}",
                f"7) {accion_2}",
            ])
            st.text_area("Resumen", resumen, height=230)

        # ========== TABLAS GERENCIALES ==========
        st.divider()

        st.markdown("### 📉 GAP por Cliente (KG)")
        by_cliente = df.groupby("Nombre de cliente", as_index=False)[["actual_kg", "budget_kg"]].sum()
        by_cliente["var_kg"] = by_cliente["actual_kg"] - by_cliente["budget_kg"]
        by_cliente["cumpl_pct"] = by_cliente.apply(
            lambda r: _pct(r["actual_kg"], r["budget_kg"]),
            axis=1
        )
        by_cliente = by_cliente.sort_values("var_kg", ascending=True)
        st.dataframe(by_cliente.head(30), use_container_width=True)

        st.markdown("### Top SKUs con mayor gap (KG)")
        by_sku = df.groupby(["ItemCode", "ItemName"], as_index=False)[["actual_kg", "budget_kg"]].sum()
        by_sku["var_kg"] = by_sku["actual_kg"] - by_sku["budget_kg"]
        by_sku = by_sku.sort_values("var_kg", ascending=True)
        st.dataframe(by_sku.head(30), use_container_width=True)

        if "Clasificación" in df.columns:
            st.markdown("### Cumplimiento por Clasificación (KG)")
            by_clas = df.groupby("Clasificación", as_index=False)[["actual_kg", "budget_kg"]].sum()
            by_clas["cumpl_pct"] = by_clas.apply(lambda r: _pct(r["actual_kg"], r["budget_kg"]), axis=1)
            by_clas = by_clas.sort_values("cumpl_pct", ascending=False)
            st.plotly_chart(px.bar(by_clas, x="Clasificación", y="cumpl_pct"), use_container_width=True)

        st.markdown("### 🧩 Pareto del déficit (clientes que explican el GAP negativo)")
        pareto_tbl = pareto_gap_clientes(df)
        if pareto_tbl.empty:
            st.info("No hay déficit (gap negativo) en el período/filtros seleccionados.")
        else:
            st.dataframe(
                pareto_tbl[["Nombre de cliente", "deficit_kg", "deficit_acum_kg", "deficit_acum_pct"]].head(30),
                use_container_width=True
            )
            chart = pareto_tbl.copy()
            chart["rank"] = range(1, len(chart) + 1)
            st.plotly_chart(px.line(chart, x="rank", y="deficit_acum_pct", markers=True), use_container_width=True)

        st.markdown("### 🧪 Controles (calidad del cruce)")
        ccol1, ccol2 = st.columns(2)

        with ccol1:
            st.markdown("**Ventas sin presupuesto (budget = 0 y actual > 0)**")
            sin_pres = df[(df["budget_kg"] == 0) & (df["actual_kg"] > 0)].copy()
            cols = [c for c in ["Nombre de cliente", "ItemCode", "ItemName", "mes", "actual_kg", "budget_kg"] if c in sin_pres.columns]
            st.dataframe(sin_pres[cols].sort_values("actual_kg", ascending=False).head(50), use_container_width=True)

        with ccol2:
            st.markdown("**Presupuesto sin ventas (actual = 0 y budget > 0)**")
            sin_ventas = df[(df["actual_kg"] == 0) & (df["budget_kg"] > 0)].copy()
            cols2 = [c for c in ["Nombre de cliente", "ItemCode", "mes", "actual_kg", "budget_kg"] if c in sin_ventas.columns]
            st.dataframe(sin_ventas[cols2].sort_values("budget_kg", ascending=False).head(50), use_container_width=True)

# --------------------- TAB 3 ---------------------
with tab3:
    st.subheader("Asistente IA técnico (basado en tu manual)")
    st.caption("Responderá basado en tu manual. Si no hay evidencia, pedirá datos faltantes (no inventa).")

    st.info(
        "Para activarlo agrega OPENAI_API_KEY y OPENAI_VECTOR_STORE_ID en Streamlit Secrets. "
        "Si no están, esta sección queda desactivada sin fallar."
    )

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
