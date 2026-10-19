"""Dashboard Streamlit – Tarea 1: Noticias peruanas, del titular al clasificador.

Ejecutar desde la carpeta tarea_1/:
    streamlit run app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # descarga HTTP clásica si el modelo viene de Hugging Face Hub

BASE = Path(__file__).parent
CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {"politica": "Política", "economia": "Economía", "deportes": "Deportes",
           "ciencia_tecnologia": "Ciencia y tecnología"}
# Carpeta local con el modelo entrenado en 02_clasificacion.ipynb, o un repo de Hugging Face Hub
MODELO = os.environ.get("MODELO_CLASIFICADOR", str(BASE / "modelo"))

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🗞️", layout="wide")


@st.cache_data
def cargar_datos():
    noticias = pd.read_csv(BASE / "datos" / "noticias.csv")
    llm = pd.read_csv(BASE / "datos" / "etiquetas_llm.csv")[["url", "etiqueta_llm", "confianza"]]
    return noticias.merge(llm, on="url", how="left")


@st.cache_data
def cargar_metricas():
    ruta = BASE / "modelo" / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


@st.cache_resource(show_spinner="Cargando el modelo DistilBERT…")
def cargar_clasificador():
    from transformers import pipeline
    return pipeline("text-classification", model=MODELO, tokenizer=MODELO, top_k=None, device=-1)


def modelo_disponible():
    local = Path(MODELO)
    return not local.exists() or any(local.glob("*.safetensors")) or any(local.glob("*.bin"))


df = cargar_datos()
metricas = cargar_metricas()

# ---------------------------------------------------------------- Encabezado
st.title("🗞️ ¿Pueden convertir titulares peruanos en un clasificador de noticias?")
st.markdown(
    "Corpus comparado de **La República** y **El Comercio** → etiquetado débil con un **LLM local** "
    "(`qwen2.5:7b` vía Ollama) → **fine-tuning de DistilBERT multilingüe** → evaluación sobre un test reservado."
)
with st.expander("📋 Resumen metodológico"):
    st.markdown(f"""
1. **Recolección:** {len(df)} noticias ({df['medio'].value_counts().to_dict()}) obtenidas de RSS oficiales,
   portadas de sección y sitemaps, respetando `robots.txt`, con pausas de 1–2 s entre solicitudes.
2. **Etiqueta de referencia:** sección original del diario mapeada a 4 categorías
   (`politica`, `economia`, `deportes`, `ciencia_tecnologia`).
3. **Línea base zero-shot:** `{metricas.get('modelo_zero_shot', 'Recognai/bert-base-spanish-wwm-cased-xnli')}` sobre 40 titulares de test.
4. **Supervisión débil:** el LLM etiqueta todo el dataset; una persona audita 20 casos al azar.
5. **Fine-tuning:** `{metricas.get('modelo_base', 'distilbert-base-multilingual-cased')}` entrenado con las etiquetas del LLM
   (split estratificado 70/15/15, `random_state=42`) y evaluado contra la sección del diario en el test.
""")

# ---------------------------------------------------------------- Filtros
st.sidebar.header("🔎 Filtros")
medios = st.sidebar.multiselect("Medio", sorted(df["medio"].unique()), default=sorted(df["medio"].unique()))
categorias = st.sidebar.multiselect("Categoría", CATEGORIAS, default=CATEGORIAS, format_func=NOMBRES.get)
filtrado = df[df["medio"].isin(medios) & df["etiqueta"].isin(categorias)]
st.sidebar.caption(f"{len(filtrado)} de {len(df)} noticias seleccionadas")

# ---------------------------------------------------------------- Indicadores
c1, c2, c3, c4 = st.columns(4)
c1.metric("Noticias recolectadas", f"{len(df)}", f"{len(filtrado)} con los filtros actuales", delta_color="off")
if metricas:
    c2.metric("F1 macro en test (DistilBERT)", f"{metricas['test_f1_macro']:.2f}",
              f"{metricas['test_f1_macro'] - metricas['zero_shot_f1_macro']:+.2f} vs. zero-shot")
    if metricas.get("auditoria_acuerdo_pct") is not None:
        c3.metric("Concordancia LLM vs. auditoría humana", f"{metricas['auditoria_acuerdo_pct']:.0f}%",
                  f"{metricas['auditoria_acuerdos']}/{metricas['auditoria_total']} casos", delta_color="off")
    else:
        c3.metric("Concordancia LLM vs. auditoría humana", "pendiente")
    c4.metric("F1 macro zero-shot (línea base)", f"{metricas['zero_shot_f1_macro']:.2f}")
else:
    c2.info("Ejecuta 02_clasificacion.ipynb para generar modelo/metricas.json")

# ---------------------------------------------------------------- Visualizaciones
col_g, col_m = st.columns([3, 2])
with col_g:
    conteo = filtrado.groupby(["etiqueta", "medio"]).size().reset_index(name="noticias")
    conteo["categoría"] = conteo["etiqueta"].map(NOMBRES)
    fig = px.bar(conteo, x="categoría", y="noticias", color="medio", barmode="group", text="noticias",
                 category_orders={"categoría": [NOMBRES[c] for c in CATEGORIAS]},
                 title="Distribución de noticias por categoría y medio (según filtros)")
    st.plotly_chart(fig, width="stretch")
with col_m:
    if metricas.get("matriz_confusion"):
        etiquetas = [NOMBRES[c] for c in CATEGORIAS]
        fig_cm = px.imshow(metricas["matriz_confusion"], x=etiquetas, y=etiquetas, text_auto=True,
                           color_continuous_scale="Blues", title="Matriz de confusión en test",
                           labels={"x": "Predicción", "y": "Real", "color": "Noticias"})
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cm, width="stretch")

# ---------------------------------------------------------------- Tabla
st.subheader("📄 Explorar las noticias")
columnas = ["titular", "bajada", "medio", "seccion_original", "etiqueta", "etiqueta_llm", "confianza", "fecha", "url"]
st.dataframe(filtrado[columnas], width="stretch", hide_index=True,
             column_config={"url": st.column_config.LinkColumn("url"),
                            "confianza": st.column_config.NumberColumn("confianza LLM", format="%.2f")})
st.download_button("⬇️ Descargar selección en CSV", filtrado[columnas].to_csv(index=False).encode("utf-8"),
                   file_name="noticias_filtradas.csv", mime="text/csv")

# ---------------------------------------------------------------- Probador en vivo
st.subheader("🧪 Probador en vivo")
st.caption("Escribe o pega un titular nuevo y el modelo DistilBERT ajustado predice su categoría.")
ejemplos = ["Congreso aprueba nuevo retiro de fondos de AFP",
            "Universitario vence a Alianza Lima en el clásico y se acerca al título",
            "NASA confirma hallazgo de agua en un asteroide cercano a la Tierra",
            "BCR mantiene su tasa de referencia en 4.25% por tercer mes consecutivo"]
ejemplo = st.selectbox("…o elige un ejemplo", [""] + ejemplos)
titular = st.text_area("Titular", value=ejemplo, height=80, placeholder="Ej.: Ministro de Economía anuncia ...")

if st.button("Clasificar", type="primary"):
    if not titular.strip():
        st.warning("Escribe un titular primero.")
    elif not modelo_disponible():
        st.error("No se encontraron los pesos del modelo en `modelo/`. Ejecuta `02_clasificacion.ipynb` "
                 "o define la variable de entorno `MODELO_CLASIFICADOR` con un repo de Hugging Face.")
    else:
        salida = cargar_clasificador()(titular.strip())
        salida = salida[0] if isinstance(salida[0], list) else salida
        probs = pd.DataFrame(salida).rename(columns={"label": "etiqueta", "score": "probabilidad"})
        probs["categoría"] = probs["etiqueta"].map(NOMBRES)
        mejor = probs.iloc[probs["probabilidad"].idxmax()]
        r1, r2 = st.columns([1, 2])
        r1.metric("Categoría predicha", mejor["categoría"], f"{mejor['probabilidad']:.1%} de probabilidad",
                  delta_color="off")
        if mejor["probabilidad"] < 0.5:
            r1.warning("Confianza baja: el titular podría pertenecer a otra categoría o a ninguna de las 4.")
        fig_p = px.bar(probs.sort_values("probabilidad"), x="probabilidad", y="categoría", orientation="h",
                       text=probs.sort_values("probabilidad")["probabilidad"].map("{:.1%}".format),
                       range_x=[0, 1], title="Probabilidad por categoría")
        r2.plotly_chart(fig_p, width="stretch")

st.divider()
st.caption("Tarea 1 – Diplomatura Python Intermedio · Datos: larepublica.pe y elcomercio.pe (uso académico).")
