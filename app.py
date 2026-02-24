import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import plotly.express as px


# ===================== CONFIG =====================
st.set_page_config(page_title="Dashboard Ventas vs Presupuesto (KG)", layout="wide")

MESES_ORDEN = [
    "Enero","Febrero","Marzo","Abril","Mayo","Junio",
    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"
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

# Manual local dentro del repo (ya lo subiste)
MANUAL_PATH = os.path.join("manual_tecnico", "Manual_tecnico_preventa.pdf")


# ===================== HELPERS NUM =====================
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _pct(a: float, b: float) -> float:
    return (a / b * 100) if b and b != 0 else 0.0


# ===================== NORMALIZACIÓN =====================
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
    long = long.rename(columns={"Nombre de cliente/proveedor": "Nombre de cliente"})
    return long

def normalizar_presupuesto(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """
    1 fila por (anio, mes, cliente, sku) con budget_kg.
    Mínimo: Nombre de cliente + ItemCode + ENE..DIC
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
    merged = ventas_long.merge(
        pres_long,
        on=["anio", "mes", "ItemCode", "Nombre de cliente"],
        how="left",
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

def ultimo_mes_con_ventas(df_filtrado: pd.DataFrame) -> Optional[str]:
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


# ===================== CONCLUSIONES (DASHBOARD) =====================
def generar_conclusiones(
    df_ytd: pd.DataFrame,
    df_anual: pd.DataFrame,
    df_periodo: pd.DataFrame,
    ultimo_mes: str | None,
    top_n: int = 5
) -> dict:
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

    pareto = pareto_gap_clientes(df_periodo)
    pareto_msg = None
    if not pareto.empty:
        n80 = int((pareto["deficit_acum_pct"] <= 80).sum())
        n80 = min(n80 + 1, len(pareto))
        total_def = float(pareto["deficit_kg"].sum())
        pareto_msg = f"Enfoque Pareto: ~{n80} clientes explican ~80% del déficit (déficit total {total_def:,.0f} KG)."

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
        recomendaciones.append("Seguimiento **semanal**: cerrar brechas en clientes deficitarios para asegurar cierre ≥100%.")

    if not clientes_deficit.empty:
        top3 = ", ".join([str(x) for x in clientes_deficit["Nombre de cliente"].head(3).tolist()])
        recomendaciones.append(f"Priorizar gestión en **clientes con déficit** (ej.: {top3}).")

    if pareto_msg:
        recomendaciones.append(pareto_msg)

    riesgos = [
        "El cruce requiere consistencia de nombres de cliente e ItemCode entre Ventas y Presupuesto.",
        "Si hay ventas sin presupuesto o presupuesto sin ventas, el cumplimiento por cliente puede sesgarse.",
    ]

    return {"conclusiones": conclusiones, "recomendaciones": recomendaciones, "riesgos": riesgos, "semaforo": semaforo}


# ===================== ASISTENTE PREVENTA OFFLINE (PDF + REGLAS) =====================
@st.cache_data(show_spinner=False)
def leer_pdf_texto(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        txt = []
        for p in reader.pages:
            t = p.extract_text() or ""
            txt.append(t)
        return "\n".join(txt).strip()
    except Exception:
        return ""

def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-záéíóúñ0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split(" ")
    # quita tokens muy cortos
    return [t for t in toks if len(t) >= 3]

def buscar_en_manual(manual: str, query: str, top_k: int = 3) -> List[str]:
    """
    Búsqueda simple por relevancia (overlap de tokens).
    Devuelve top_k fragmentos (paragraphs) del manual.
    """
    if not manual.strip():
        return []
    q_toks = set(_tokenize(query))
    if not q_toks:
        return []

    # dividir en fragmentos (párrafos)
    parts = [p.strip() for p in manual.split("\n") if p.strip()]
    # agrupar líneas cercanas para tener fragmentos más útiles
    chunks = []
    buf = []
    for line in parts:
        buf.append(line)
        if len(" ".join(buf)) > 500:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))

    scored: List[Tuple[int, str]] = []
    for c in chunks:
        c_toks = set(_tokenize(c))
        score = len(q_toks.intersection(c_toks))
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

@dataclass
class SpecsInput:
    producto: Optional[str] = None
    peso_g: Optional[int] = None
    vida_meses: Optional[int] = None
    maquina: Optional[str] = None  # VFFS/HFFS
    formato: Optional[str] = None  # bolsa/bobina
    grasa: Optional[bool] = None
    humedad: Optional[bool] = None
    quiere_bajar_micras: bool = False

def parse_pregunta(q: str) -> SpecsInput:
    s = q.lower()

    # producto (heurístico)
    producto = None
    for key in ["café", "cafe", "snack", "detergente", "congel", "congelado", "congelados"]:
        if key in s:
            if "cafe" in key:
                producto = "café"
            elif "snack" in key:
                producto = "snacks"
            elif "detergente" in key:
                producto = "detergente"
            elif "congel" in key:
                producto = "congelados"
            break

    # peso en g/kg
    peso_g = None
    mkg = re.search(r"(\d+(?:\.\d+)?)\s*kg", s)
    if mkg:
        try:
            peso_g = int(float(mkg.group(1)) * 1000)
        except:
            pass
    mg = re.search(r"(\d+)\s*g", s)
    if mg and not peso_g:
        try:
            peso_g = int(mg.group(1))
        except:
            pass

    # vida útil
    vida_meses = None
    mm = re.search(r"(\d+)\s*mes", s)
    if mm:
        try:
            vida_meses = int(mm.group(1))
        except:
            pass
    ma = re.search(r"(\d+)\s*año", s)
    if ma:
        try:
            vida_meses = int(ma.group(1)) * 12
        except:
            pass

    # máquina
    maquina = None
    if "vffs" in s:
        maquina = "VFFS"
    elif "hffs" in s:
        maquina = "HFFS"

    # formato
    formato = None
    if "bobina" in s:
        formato = "bobina"
    elif "bolsa" in s:
        formato = "bolsa"

    # grasa/humedad
    grasa = None
    if "graso" in s or "grasa" in s:
        grasa = True
    if "sin grasa" in s or "no graso" in s:
        grasa = False

    humedad = None
    if "humedad" in s or "higros" in s:
        humedad = True

    quiere_bajar_micras = any(x in s for x in ["bajar micra", "reducir micra", "bajar calibre", "reducir calibre", "bajar micras", "reducir micras"])

    return SpecsInput(
        producto=producto,
        peso_g=peso_g,
        vida_meses=vida_meses,
        maquina=maquina,
        formato=formato,
        grasa=grasa,
        humedad=humedad,
        quiere_bajar_micras=quiere_bajar_micras
    )

def sugerir_micras_por_peso(peso_g: int) -> Tuple[int, int]:
    """
    Devuelve rango micras sugeridas (min,max) según manual.
    """
    if peso_g <= 100:
        return (60, 70)
    if peso_g <= 250:
        return (70, 80)
    if peso_g <= 500:
        return (80, 90)
    if peso_g <= 1000:
        return (90, 110)
    return (110, 150)

def margen_seguridad(spec: SpecsInput) -> int:
    """
    Margen seguridad simple.
    """
    add = 0
    # Alta velocidad no la detectamos; aquí usamos máquina y peso como proxy
    if spec.peso_g and spec.peso_g >= 1000:
        add += 5
    if spec.maquina == "VFFS":
        add += 5
    if spec.vida_meses and spec.vida_meses > 12:
        add += 5
    if spec.formato == "bolsa":
        add += 0
    return add

def recomendacion_por_producto(spec: SpecsInput) -> Dict[str, str]:
    """
    Reglas basadas en tu manual.
    Devuelve estructuras base A (segura) y B (costo) si aplica.
    """
    prod = spec.producto or ""
    if prod == "café":
        return {
            "A": "PET12 / AL7 / PE70 (alta barrera aroma)",
            "B": "PET12 / METPET12 / PE70 (barrera media-alta, menor costo)"
        }
    if prod == "snacks":
        return {
            "A": "BOPP20 / METBOPP20 / CPP30 (snack graso, barrera media)",
            "B": "PET12 / METPET12 / PE60 (alternativa costo según requerimiento)"
        }
    if prod == "detergente":
        return {
            "A": "PET12 / PE80–100 (resistencia + sellado)",
            "B": "BOPP25 / PE90 (alternativa costo si máquina lo permite)"
        }
    if prod == "congelados":
        return {
            "A": "PET12 / PE100–110 (congelados, resistencia)",
            "B": "BOPP20 / PE90 (alternativa costo si cumple performance)"
        }
    # genérico
    return {
        "A": "Estructura laminada con capa de sellado PE/CPP según máquina y producto",
        "B": "Versión metalizada (METPET/METBOPP) si se requiere barrera y se busca optimizar costo"
    }

def validar_datos_minimos(spec: SpecsInput) -> List[str]:
    faltan = []
    if not spec.producto:
        faltan.append("Producto (qué empacas: café/snack/detergente/congelado u otro)")
    if not spec.peso_g:
        faltan.append("Peso por unidad (g o kg)")
    if not spec.vida_meses:
        faltan.append("Vida útil requerida (meses)")
    if not spec.maquina:
        faltan.append("Tipo de máquina (VFFS o HFFS)")
    if not spec.formato:
        faltan.append("Formato (bolsa o bobina)")
    return faltan

def construir_respuesta_preventa(manual: str, pregunta: str) -> str:
    spec = parse_pregunta(pregunta)
    faltan = validar_datos_minimos(spec)

    evidencias = buscar_en_manual(manual, pregunta, top_k=3)

    # Si falta info crítica, pedirla (no inventar)
    if faltan:
        resp = []
        resp.append("## 7) Datos faltantes (obligatorio para especificación final)")
        for f in faltan:
            resp.append(f"- {f}")
        resp.append("\n---\n## 8) Evidencia del manual (fragmentos relacionados)")
        if evidencias:
            for i, e in enumerate(evidencias, 1):
                resp.append(f"**Fragmento {i}:** {e[:900]}{'...' if len(e) > 900 else ''}")
        else:
            resp.append("_No se encontró fragmento relevante en el manual para la pregunta exacta. Recomiendo completar checklist._")
        return "\n".join(resp)

    # Recomendaciones
    mic_min, mic_max = sugerir_micras_por_peso(spec.peso_g or 0)
    add = margen_seguridad(spec)
    mic_sugerida_min = mic_min + add
    mic_sugerida_max = mic_max + add

    estructuras = recomendacion_por_producto(spec)

    # Riesgos (preventivos)
    riesgos = []
    if spec.vida_meses and spec.vida_meses > 12 and spec.producto != "café":
        riesgos.append("Vida útil > 12 meses: validar barrera. Considerar aluminio si hay sensibilidad alta.")
    if spec.peso_g and spec.peso_g > 1000 and mic_sugerida_min < 110:
        riesgos.append("Peso > 1kg: evitar micras bajas. Riesgo de falla mecánica/sellos.")
    if spec.maquina == "VFFS":
        riesgos.append("VFFS: asegurar buen hot tack y ventana de sellado. Evitar estructura demasiado rígida.")
    if spec.quiere_bajar_micras:
        riesgos.append("Solicitud de bajar micras: aumenta riesgo de fuga, perforación y reclamo. Requiere prueba piloto.")

    # Nota producción (estándar)
    pruebas = [
        "Validar ventana de sellado (set up).",
        "Prueba de sellado / fuga.",
        "Prueba de caída si peso ≥ 500g.",
        "Revisión registro de impresión (si aplica).",
    ]

    # Construcción respuesta final
    out = []
    out.append("## 1) Recomendación técnica base (Opción A — segura)")
    out.append(f"- **Estructura:** {estructuras['A']}")
    out.append(f"- **Micras sugeridas:** {mic_sugerida_min}–{mic_sugerida_max} µ (incluye margen seguridad)")
    out.append(f"- **Contexto detectado:** producto={spec.producto}, peso={spec.peso_g}g, vida útil={spec.vida_meses} meses, máquina={spec.maquina}, formato={spec.formato}")

    out.append("\n## 2) Alternativa optimizada costo (Opción B)")
    out.append(f"- **Estructura:** {estructuras['B']}")
    out.append(f"- **Micras sugeridas:** {max(mic_min, mic_sugerida_min - 10)}–{max(mic_max, mic_sugerida_max - 10)} µ (solo si pruebas confirman)")

    out.append("\n## 3) Margen de seguridad aplicado (y por qué)")
    out.append(f"- **+{add} µ** por riesgo operativo (peso/máquina/vida útil según manual).")

    out.append("\n## 4) Riesgo técnico (posibles fallas/reclamos)")
    if riesgos:
        for r in riesgos:
            out.append(f"- {r}")
    else:
        out.append("- Riesgo bajo con la información actual; confirmar pruebas estándar de sellado.")

    out.append("\n## 5) Impacto comercial (vida útil / reclamo / negociación)")
    if spec.quiere_bajar_micras:
        out.append("- Recomiendo **ofrecer B como alternativa** con condiciones: prueba piloto y validación de sellado. No bajar a ciegas para evitar reclamo.")
    out.append("- Presentar A como opción robusta (menor reclamo) y B como opción costo (condicionada a pruebas).")

    out.append("\n## 6) Nota para producción (pruebas requeridas)")
    for p in pruebas:
        out.append(f"- {p}")

    out.append("\n## 7) Datos faltantes (si aplica)")
    out.append("- Ninguno crítico detectado. Si hay restricciones de máquina (ancho, core, diámetro), agregarlas.")

    out.append("\n---\n## 8) Evidencia del manual (fragmentos relacionados)")
    if evidencias:
        for i, e in enumerate(evidencias, 1):
            out.append(f"**Fragmento {i}:** {e[:900]}{'...' if len(e) > 900 else ''}")
    else:
        out.append("_No se encontró fragmento relevante en el manual para la pregunta exacta. Revisa el manual o amplía la consulta._")

    return "\n".join(out)


# ===================== UI =====================
st.title("📊 Dashboard Gerencial — Cumplimiento vs Presupuesto (KG)")
tab1, tab2, tab3 = st.tabs(["1) Cargar Excel", "2) Dashboard (KG)", "3) Asistente Preventa (PDF local)"])


# -------- TAB 1: CARGA --------
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

            except Exception as e:
                st.exception(e)


# -------- TAB 2: DASHBOARD --------
with tab2:
    st.subheader("Cumplimiento vs Presupuesto (KG)")

    if "df_final" not in st.session_state:
        st.warning("Primero carga y procesa tus Excel en la pestaña 1.")
    else:
        df_all = st.session_state["df_final"].copy()

        st.sidebar.header("Filtros")
        anio = st.sidebar.selectbox("Año", sorted(df_all["anio"].unique()))
        df_all = df_all[df_all["anio"] == anio]

        # Vendedor (solo si hay más de 1)
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
                sem = "🟢 Verde"
            elif proy_pct >= 95:
                sem = "🟡 Amarillo"
            else:
                sem = "🔴 Rojo"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Actual YTD (KG) | hasta {ultimo_mes}", f"{actual_ytd:,.0f}")
            c2.metric("Budget YTD (KG)", f"{budget_ytd:,.0f}")
            c3.metric("Varianza YTD (KG)", f"{var_ytd:,.0f}")
            c4.metric("% Cumplimiento YTD", f"{cumpl_ytd:.1f}%")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Run Rate (KG/mes)", f"{run_rate:,.0f}")
            c6.metric("Proyección anual (KG)", f"{proyeccion_anual:,.0f}")
            c7.metric("Proyección vs Presupuesto anual", f"{proy_pct:.1f}%")
            c8.metric("Semáforo", sem)

            st.caption(f"KG necesarios/mes para cumplir meta anual: {kg_necesarios_mes:,.0f} (meses restantes: {meses_restantes})")

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


# -------- TAB 3: ASISTENTE PREVENTA OFFLINE --------
with tab3:
    st.subheader("Asistente Preventa — basado en PDF local (sin API)")
    st.caption("Responde usando tu manual en el repositorio. Si faltan datos críticos, los pedirá. No inventa.")

    if not os.path.exists(MANUAL_PATH):
        st.error(f"No encuentro el PDF del manual en: `{MANUAL_PATH}`")
        st.info("Verifica que exista en GitHub: manual_tecnico/Manual_tecnico_preventa.pdf (respetando mayúsculas/minúsculas).")
        st.stop()

    manual_text = leer_pdf_texto(MANUAL_PATH)
    if not manual_text:
        st.warning("Pude abrir el PDF, pero no pude extraer texto. Puede ser un PDF escaneado (imagen).")
        st.info("Solución: exporta el manual desde Google Docs como PDF (texto) y vuelve a subirlo al repo.")
        st.stop()

    with st.expander("📄 Estado del manual"):
        st.write(f"Ruta: {MANUAL_PATH}")
        st.write(f"Texto extraído: {len(manual_text):,} caracteres")
        st.caption("Si el texto es muy corto, revisa que el PDF no sea escaneado.")

    if "chat_preventa_off" not in st.session_state:
        st.session_state["chat_preventa_off"] = []

    for msg in st.session_state["chat_preventa_off"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ej: Snack 250g, VFFS, vida útil 6 meses. Cliente quiere bajar micras. ¿Qué ofrezco?")

    if user_q:
        st.session_state["chat_preventa_off"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            answer = construir_respuesta_preventa(manual_text, user_q)
            st.markdown(answer)
            st.session_state["chat_preventa_off"].append({"role": "assistant", "content": answer})
