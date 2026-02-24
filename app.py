import os
import pandas as pd
import streamlit as st
import plotly.express as px

# ========= CONFIG =========
st.set_page_config(page_title="Dashboard Ventas vs Presupuesto (KG)", layout="wide")

MESES_ORDEN = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
]

MESES_VENTAS = {
    "Enero": "Ene_KG", "Febrero": "Feb_KG", "Marzo": "Mar_KG", "Abril": "Abr_KG",
    "Mayo": "May_KG", "Junio": "Jun_KG", "Julio": "Jul_KG", "Agosto": "Ago_KG",
    "Septiembre": "Sep_KG", "Octubre": "Oct_KG", "Noviembre": "Nov_KG", "Diciembre": "Dic_KG",
}

MESES_PRES = {
    "Enero": "ENE", "Febrero": "FEB", "Marzo": "MAR", "Abril": "ABR",
    "Mayo": "MAY", "Junio": "JUN", "Julio": "JUL", "Agosto": "AGO",
    "Septiembre": "SEP", "Octubre": "OCT", "Noviembre": "NOV", "Diciembre": "DIC",
}

MANUAL_PATH = "manual_tecnico/Manual_tecnico_preventa.pdf"

# ========= HELPERS =========
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _pct(a: float, b: float) -> float:
    return (a / b * 100) if b and b != 0 else 0.0

def normalizar_ventas(df: pd.DataFrame, anio: int) -> pd.DataFrame:
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
    long = long.rename(columns={"Nombre de cliente/proveedor": "Nombre de cliente"})
    return long

def normalizar_presupuesto(df: pd.DataFrame, anio: int) -> pd.DataFrame:
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
    merged = ventas_long.merge(
        pres_long,
        on=["anio", "mes", "ItemCode", "Nombre de cliente"],
        how="left"
    )
    merged["budget_kg"] = merged["budget_kg"].fillna(0)
    merged["var_kg"] = merged["actual_kg"] - merged["budget_kg"]
    merged["cumpl_pct"] = (merged["actual_kg"] / merged["budget_kg"]).replace([float("inf")], 0).fillna(0) * 100
    return merged

def kpis(df: pd.DataFrame) -> dict:
    actual = float(df["actual_kg"].sum())
    budget = float(df["budget_kg"].sum())
    var = actual - budget
    cumpl = _pct(actual, budget)
    return {"actual": actual, "budget": budget, "var": var, "cumpl": cumpl}

def ultimo_mes_con_ventas(df_filtrado: pd.DataFrame):
    by_mes = df_filtrado.groupby("mes", as_index=False)["actual_kg"].sum().sort_values("mes")
    by_mes = by_mes[by_mes["actual_kg"] > 0]
    if by_mes.empty:
        return None
    return str(by_mes.iloc[-1]["mes"])

def pareto_gap_clientes(df: pd.DataFrame) -> pd.DataFrame:
    by_cliente = df.groupby("Nombre de cliente", as_index=False)[["actual_kg", "budget_kg"]].sum()
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

def generar_conclusiones(df_ytd: pd.DataFrame, df_anual: pd.DataFrame, df_periodo: pd.DataFrame, ultimo_mes: str | None, top_n: int = 5):
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

    actual_p = float(df_periodo["actual_kg"].sum())
    budget_p = float(df_periodo["budget_kg"].sum())
    var_p = actual_p - budget_p
    cumpl_p = _pct(actual_p, budget_p)

    by_cliente = df_periodo.groupby("Nombre de cliente", as_index=False)[["actual_kg", "budget_kg"]].sum()
    by_cliente["var_kg"] = by_cliente["actual_kg"] - by_cliente["budget_kg"]
    clientes_deficit = by_cliente[by_cliente["var_kg"] < 0].sort_values("var_kg").head(top_n)

    by_sku = df_periodo.groupby(["ItemCode", "ItemName"], as_index=False)[["actual_kg", "budget_kg"]].sum()
    by_sku["var_kg"] = by_sku["actual_kg"] - by_sku["budget_kg"]
    skus_deficit = by_sku[by_sku["var_kg"] < 0].sort_values("var_kg").head(top_n)

    pareto = pareto_gap_clientes(df_periodo)
    pareto_msg = None
    if not pareto.empty:
        n80 = int((pareto["deficit_acum_pct"] <= 80).sum())
        n80 = min(n80 + 1, len(pareto))
        total_def = float(pareto["deficit_kg"].sum())
        pareto_msg = f"Enfoque Pareto: ~{n80} clientes explican ~80% del déficit (déficit total {total_def:,.0f} KG)."

    ventas_sin_pres = df_periodo[(df_periodo["budget_kg"] == 0) & (df_periodo["actual_kg"] > 0)]
    pres_sin_ventas = df_periodo[(df_periodo["actual_kg"] == 0) & (df_periodo["budget_kg"] > 0)]
    kgs_ventas_sin_pres = float(ventas_sin_pres["actual_kg"].sum())
    kgs_pres_sin_ventas = float(pres_sin_ventas["budget_kg"].sum())

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

    riesgos = []
    if kgs_ventas_sin_pres > 0 or kgs_pres_sin_ventas > 0:
        riesgos.append("Desalineación entre presupuesto y ventas (cliente/SKU). Puede sesgar el cumplimiento por cliente.")
    riesgos.append("El cruce requiere consistencia de nombres de cliente y ItemCode entre ambos archivos.")

    return {"conclusiones": conclusiones, "recomendaciones": recomendaciones, "riesgos": riesgos, "semaforo": semaforo}

@st.cache_data(show_spinner=False)
def cargar_manual_texto(pdf_path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        texto = ""
        for page in reader.pages:
            t = page.extract_text() or ""
            texto += t + "\n"
        return texto.strip()
    except Exception:
        return ""

def leer_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return os.environ.get("OPENAI_API_KEY")

# ========= UI =========
st.title("📊 Dashboard Gerencial — Cumplimiento vs Presupuesto (KG)")
tab1, tab2, tab3 = st.tabs(["1) Cargar Excel", "2) Dashboard (KG)", "3) Asistente IA Preventa (sin Vector Store)"])

# ----- TAB 1 -----
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
                st.success("✅ Archivos procesados. Ve a la pestaña 'Dashboard (KG)'.")

            except Exception as e:
                st.exception(e)

# ----- TAB 2 -----
with tab2:
    st.subheader("Cumplimiento vs Presupuesto (KG)")
    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df_all = st.session_state["df_final"].copy()

        st.sidebar.header("Filtros")
        anio = st.sidebar.selectbox("Año", sorted(df_all["anio"].unique()))
        df_all = df_all[df_all["anio"] == anio]

        if "SlpName" in df_all.columns:
            vendedores = sorted(df_all["SlpName"].dropna().unique())
            if len(vendedores) > 1:
                vend_sel = st.sidebar.multiselect("Vendedor", vendedores, default=vendedores)
                if vend_sel:
                    df_all = df_all[df_all["SlpName"].isin(vend_sel)]

        mes_sel = st.sidebar.multiselect("Mes (para análisis)", MESES_ORDEN, default=MESES_ORDEN)

        st.markdown("## 🧭 Ejecutivo (YTD automático + Proyección)")
        ultimo_mes = ultimo_mes_con_ventas(df_all)

        if ultimo_mes is None:
            st.warning("No hay ventas (KG) con los filtros seleccionados. No se puede calcular YTD automático.")
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

            st.caption(f"KG necesarios por mes para cumplir la meta anual: {kg_necesarios_mes:,.0f} (meses restantes: {meses_restantes})")

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

        st.divider()
        st.markdown("## 🧾 Conclusiones y recomendaciones")
        if ultimo_mes is None or df_ytd.empty:
            st.info("No se generaron conclusiones porque no hay ventas > 0 para calcular YTD automático.")
        else:
            insights = generar_conclusiones(df_ytd=df_ytd, df_anual=df_all, df_periodo=df, ultimo_mes=ultimo_mes, top_n=5)

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

        st.divider()
        st.markdown("### 🧩 Pareto del déficit (clientes que explican el GAP negativo)")
        pareto_tbl = pareto_gap_clientes(df)
        if pareto_tbl.empty:
            st.info("No hay déficit (gap negativo) en el período/filtros seleccionados.")
        else:
            st.dataframe(
                pareto_tbl[["Nombre de cliente", "deficit_kg", "deficit_acum_kg", "deficit_acum_pct"]].head(30),
                use_container_width=True
            )

# ----- TAB 3 (SIN VECTOR STORE) -----
with tab3:
    st.subheader("Asistente IA — Modo Preventa Estratégico (sin Vector Store)")
    st.caption("Usa el manual en PDF del repositorio. Si no hay evidencia, pedirá datos faltantes. No inventa.")

    # 1) Cargar manual desde repo
    manual_texto = ""
    if os.path.exists(MANUAL_PATH):
        manual_texto = cargar_manual_texto(MANUAL_PATH)

    if not manual_texto:
        st.warning(
            "No se encontró el manual o no se pudo leer.\n\n"
            f"Verifica que exista en tu repo: `{MANUAL_PATH}`"
        )
        st.stop()

    with st.expander("📄 Ver estado del manual cargado"):
        st.write(f"Ruta: {MANUAL_PATH}")
        st.write(f"Caracteres leídos: {len(manual_texto):,}")
        st.caption("Si ves 0 caracteres, el PDF puede ser escaneado (imagen) y no texto. En ese caso hay que convertirlo a PDF con texto.")

    # 2) API key (opcional, pero necesaria para responder)
    api_key = leer_api_key()
    if not api_key:
        st.info(
            "Para activar el asistente, agrega tu `OPENAI_API_KEY` en Streamlit Secrets.\n\n"
            "Mientras tanto, el dashboard funciona normal."
        )

    # 3) Chat
    if "chat_preventa" not in st.session_state:
        st.session_state["chat_preventa"] = []

    for msg in st.session_state["chat_preventa"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ej: Snack 250g, VFFS, vida útil 6 meses. Cliente quiere bajar micras. ¿Qué ofrezco?")

    if user_q:
        st.session_state["chat_preventa"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            if not api_key:
                st.markdown(
                    "No puedo responder aún porque falta `OPENAI_API_KEY` en Secrets.\n\n"
                    "Cuando la agregues, podrás usar el asistente."
                )
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)

                    system_instructions = """
Eres un asistente de PREVENTA ESTRATÉGICO para empaque plástico flexible (bolsa/bobina).
Reglas obligatorias:
- Responde SOLO usando el contenido del manual que recibes abajo.
- Si el manual no tiene información suficiente, NO inventes: pide datos faltantes del checklist.
- Siempre ofrece dos opciones cuando sea viable:
  A) Opción técnica segura (menor riesgo)
  B) Opción optimizada costo (si es viable)
- Formato:
  1) Recomendación técnica base (estructura + micras + nivel barrera)
  2) Alternativa optimizada costo (estructura + micras) [si aplica]
  3) Margen de seguridad aplicado (y por qué)
  4) Riesgo técnico (probables fallas/reclamos)
  5) Impacto comercial (vida útil / reclamo / negociación)
  6) Nota para producción (pruebas requeridas)
  7) Datos faltantes (si aplica)
  8) Evidencia del manual (qué parte sustenta la respuesta)
"""

                    # Para evitar prompts gigantes, recortamos el manual si es muy largo.
                    # (Esto mejora estabilidad y costos). Ajusta si tu manual crece mucho.
                    MAX_CHARS = 25000
                    manual_for_prompt = manual_texto[:MAX_CHARS]

                    resp = client.responses.create(
                        model="gpt-4.1-mini",
                        input=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": f"MANUAL:\n{manual_for_prompt}\n\nPREGUNTA:\n{user_q}"},
                        ],
                    )

                    answer = resp.output_text
                    st.markdown(answer)
                    st.session_state["chat_preventa"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error("No se pudo conectar a OpenAI. Revisa tu API Key (puede estar inválida o sin billing).")
                    st.exception(e)
