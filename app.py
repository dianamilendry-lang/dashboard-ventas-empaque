Ese `if` quedó fuera del `with tab4:` y sin indentación correcta.
- Pegaste el prompt del asistente **sin** `prompt = f""" ... """` (por eso Python lo interpreta como código y marca `1) Opción A segura` como error).
- Estás buscando el manual en `manual_tecnico/` pero tu PDF está en la **raíz** del repo (`Manual_tecnico_preventa.pdf`).

---

# ✅ Te dejo el `app.py` COMPLETO corregido (copia/pega todo)

Este ya:
- no truena,
- detecta el PDF en raíz **o** en carpeta,
- tiene Tab 4 funcionando con Gemini,
- mantiene dashboard mensual,
- mantiene IA del dashboard.

```python
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

# Tu PDF actualmente está en la raíz del repo (según tu screenshot)
MANUAL_PATH_ROOT = "Manual_tecnico_preventa.pdf"
MANUAL_DIR = "manual_tecnico"

# ===================== GEMINI =====================
def gemini_client():
  api_key = st.secrets.get("GEMINI_API_KEY", None)
  if not api_key:
      st.error("Falta GEMINI_API_KEY en Secrets (Streamlit Cloud).")
      st.stop()
  return genai.Client(api_key=api_key)

def gemini_generate(prompt: str) -> str:
  client = gemini_client()
  resp = client.models.generate_content(
      model="gemini-2.0-flash",
      contents=prompt
  )
  return resp.text or "(Sin respuesta)"

# ===================== MANUAL: detectar + leer =====================
@st.cache_data(show_spinner=False)
def find_manual_pdf() -> str | None:
  # 1) Primero intenta en la raíz
  if os.path.exists(MANUAL_PATH_ROOT):
      return MANUAL_PATH_ROOT

  # 2) Si no está, intenta dentro de manual_tecnico/
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

# ===================== RAG SIMPLE (token overlap) =====================
def tokenize(text: str) -> set:
  text = re.sub(r"[^a-zA-Z0-9áéíóúñ ]", " ", text.lower())
  text = re.sub(r"\s+", " ", text).strip()
  return set([t for t in text.split(" ") if len(t) >= 3])

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
  # Chunk por caracteres, robusto con PDFs
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

# ===================== NORMALIZACIÓN MENSUAL =====================
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

# ===================== UI =====================
st.title("📊 Dashboard Mensual + IA Generativa")

tab1, tab2, tab3, tab4 = st.tabs([
  "1) Cargar Excel",
  "2) Dashboard",
  "3) IA del Dashboard",
  "4) IA Técnica Preventa"
])

# -------- TAB 1: CARGA --------
with tab1:
  ventas_file = st.file_uploader("Reporte Ventas mensual (.xlsx)", type=["xlsx"])
  pres_file = st.file_uploader("Presupuesto mensual (.xlsx)", type=["xlsx"])

  if st.button("Procesar"):
      if not ventas_file or not pres_file:
          st.error("Sube ambos archivos.")
      else:
          try:
              dfv = pd.read_excel(ventas_file)
              dfp = pd.read_excel(pres_file)

              ventas_long = normalizar_ventas(dfv)
              pres_long = normalizar_pres(dfp)

              df = ventas_long.merge(pres_long, on=["ItemCode", "mes"], how="left")
              df["budget_kg"] = df["budget_kg"].fillna(0)
              df["var_kg"] = df["actual_kg"] - df["budget_kg"]
              df["cumpl_pct"] = (df["actual_kg"] / df["budget_kg"]).replace([float("inf")], 0).fillna(0) * 100

              st.session_state["df"] = df
              st.success("✅ Datos procesados. Ve al Dashboard.")
          except Exception as e:
              st.exception(e)

# -------- TAB 2: DASHBOARD --------
with tab2:
  if "df" not in st.session_state:
      st.warning("Carga datos primero en la pestaña 1.")
  else:
      df = st.session_state["df"]

      total_actual = float(df["actual_kg"].sum())
      total_budget = float(df["budget_kg"].sum())
      total_var = total_actual - total_budget
      total_pct = (total_actual / total_budget * 100) if total_budget > 0 else 0.0

      c1, c2, c3, c4 = st.columns(4)
      c1.metric("Actual (KG)", f"{total_actual:,.0f}")
      c2.metric("Budget (KG)", f"{total_budget:,.0f}")
      c3.metric("Varianza (KG)", f"{total_var:,.0f}")
      c4.metric("% Cumplimiento", f"{total_pct:.1f}%")

      by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg"]].sum().sort_values("mes")
      st.plotly_chart(px.line(by_mes, x="mes", y=["actual_kg", "budget_kg"], markers=True),
                      use_container_width=True)

# -------- TAB 3: IA DEL DASHBOARD --------
with tab3:
  if "df" not in st.session_state:
      st.warning("Carga datos primero en la pestaña 1.")
  else:
      df = st.session_state["df"]

      by_mes = df.groupby("mes", as_index=False)[["actual_kg", "budget_kg", "var_kg"]].sum().sort_values("mes")
      resumen = by_mes.to_string(index=False)

      if st.button("Generar Análisis Ejecutivo con IA"):
          prompt = f"""
Eres analista financiero industrial (empaque flexible).
Reglas:
- Usa SOLO los números proporcionados.
- No inventes cifras.
- Si falta información para recomendar, pide el dato faltante.

Datos (por mes):
{resumen}

Entrega:
1) Resumen ejecutivo (5 bullets)
2) Conclusiones clave (causas probables, sin inventar)
3) Recomendaciones comerciales (acciones concretas)
4) Riesgos y supuestos críticos
"""
          try:
              response = gemini_generate(prompt)
              st.markdown(response)
          except Exception as e:
              st.error("Error llamando a Gemini. Revisa GEMINI_API_KEY y permisos.")
              st.exception(e)

# -------- TAB 4: IA TÉCNICA PREVENTA --------
with tab4:
  st.subheader("IA Técnica Preventa (Gemini + Manual)")

  pdf_path = find_manual_pdf()
  if not pdf_path:
      st.warning("No encuentro el PDF del manual. Súbelo al repo como 'Manual_tecnico_preventa.pdf' en la raíz o dentro de 'manual_tecnico/'.")
      st.stop()

  st.caption(f"Manual detectado: `{pdf_path}`")

  manual_text = load_manual_text(pdf_path)
  if not manual_text:
      st.warning("Pude abrir el PDF, pero no pude extraer texto. Si es escaneado, expórtalo como PDF con texto.")
      st.stop()

  chunks = chunk_text(manual_text)

  if "chat_tecnico" not in st.session_state:
      st.session_state["chat_tecnico"] = []

  for m in st.session_state["chat_tecnico"]:
      with st.chat_message(m["role"]):
          st.markdown(m["content"])

  pregunta = st.chat_input("Ej: Café 500g, VFFS, vida útil 12 meses. ¿Estructura y micras?")

  if pregunta:
      st.session_state["chat_tecnico"].append({"role": "user", "content": pregunta})
      with st.chat_message("user"):
          st.markdown(pregunta)

      context = retrieve_chunks(chunks, pregunta, top_k=4)
      contexto_txt = "\n\n---\n\n".join([f"[Fragmento {i+1}]\n{c}" for i, c in enumerate(context)])

      prompt = f"""
Eres ingeniero preventa en empaque plástico flexible (bolsa y bobina).

REGLAS:
- Responde SOLO usando el CONTEXTO del manual.
- Si no hay evidencia suficiente, NO inventes: pide datos faltantes (producto, peso, vida útil, máquina VFFS/HFFS, barrera, calibre, tipo de sello, etc.).
- Da 2 opciones si aplica:
A) segura (conservadora)
B) optimizada costo (condicionada a prueba)

FORMATO:
1) Opción A segura
2) Opción B optimizada costo (si aplica)
3) Datos faltantes / supuestos
4) Riesgos técnicos y comerciales
5) Evidencia usada (menciona fragmentos)

CONTEXTO (manual):
{contexto_txt if contexto_txt else "SIN CONTEXTO RELEVANTE ENCONTRADO."}

PREGUNTA:
{pregunta}
"""
      try:
          respuesta = gemini_generate(prompt)
      except Exception as e:
          respuesta = "Error llamando a Gemini. Revisa GEMINI_API_KEY en Secrets."
          st.exception(e)

      with st.chat_message("assistant"):
          st.markdown(respuesta)

      st.session_state["chat_tecnico"].append({"role": "assistant", "content": respuesta})
