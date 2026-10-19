"""Dashboard Streamlit – Tarea 1: Noticias peruanas, del titular al clasificador.

Ejecutar desde la carpeta tarea_1/:
    streamlit run app.py
"""
import json
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # descarga HTTP clásica si el modelo viene de Hugging Face Hub

BASE = Path(__file__).parent
# Carpeta local con el modelo entrenado en 02_clasificacion.ipynb, o un repo de Hugging Face Hub
MODELO = os.environ.get("MODELO_CLASIFICADOR", str(BASE / "modelo"))
LIMA = timezone(timedelta(hours=-5))

CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {"politica": "Política", "economia": "Economía", "deportes": "Deportes",
           "ciencia_tecnologia": "Ciencia y tecnología"}
ORDEN = [NOMBRES[c] for c in CATEGORIAS]
# Paleta categórica de referencia (posiciones 1-4, orden fijo): el color sigue a la categoría en toda la app
COLOR_CAT = {"Política": "#2a78d6", "Economía": "#eb6834", "Deportes": "#1baf7a", "Ciencia y tecnología": "#eda100"}
MEDIOS = {"La Republica": "La República", "El Comercio": "El Comercio"}
GRIS, DESTACADO = "#b4b8bf", "#2a78d6"

# Mismo mapeo sección → categoría que 01_recoleccion.ipynb
MAPA_SECCIONES = {"politica": "politica", "economia": "economia", "deportes": "deportes",
                  "deporte-total": "deportes", "ciencia": "ciencia_tecnologia", "tecnologia": "ciencia_tecnologia"}
# RSS oficiales (permitidos en robots.txt): portada + las 4 secciones de cada medio
_EC = "https://elcomercio.pe/arc/outboundfeeds/rss"
FEEDS = [("La República", "https://larepublica.pe/rss/home.xml")] + \
        [("La República", f"https://larepublica.pe/rss/{s}.xml") for s in ["politica", "economia", "deportes", "ciencia"]] + \
        [("El Comercio", f"{_EC}/?outputType=xml")] + \
        [("El Comercio", f"{_EC}/category/{s}/?outputType=xml") for s in ["politica", "economia", "deporte-total", "tecnologia"]]
USER_AGENT = "TareaNLP-Diplomatura-PythonIntermedio/1.0 (dashboard academico; lectura de RSS cada 30 min, 1-2 s entre solicitudes)"

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🏷️", layout="centered")


# ================================================================ Datos y modelo
@st.cache_data
def cargar_datos():
    noticias = pd.read_csv(BASE / "datos" / "noticias.csv")
    llm = pd.read_csv(BASE / "datos" / "etiquetas_llm.csv")[["url", "etiqueta_llm", "confianza"]]
    df = noticias.merge(llm, on="url", how="left")
    df["categoria"] = df["etiqueta"].map(NOMBRES)
    df["categoria_llm"] = df["etiqueta_llm"].map(NOMBRES)
    df["medio_nombre"] = df["medio"].map(MEDIOS)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


@st.cache_data
def cargar_metricas():
    ruta = BASE / "modelo" / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def modelo_disponible():
    local = Path(MODELO)
    return not local.exists() or any(local.glob("*.safetensors")) or any(local.glob("*.bin"))


@st.cache_resource(show_spinner="Cargando el modelo DistilBERT…")
def cargar_clasificador():
    from transformers import pipeline
    return pipeline("text-classification", model=MODELO, tokenizer=MODELO, top_k=None, device=-1)


def clasificar(titulares):
    """Devuelve una lista de dicts {categoría: probabilidad}, uno por titular."""
    salidas = cargar_clasificador()(list(titulares), batch_size=16)
    return [{NOMBRES[d["label"]]: d["score"] for d in s} for s in salidas]


def seccion_y_categoria(url):
    """Sección según la URL (mismas reglas que 01_recoleccion.ipynb) y su categoría, o None si no es de las 4."""
    partes = [p for p in urlparse(url).path.split("/") if p]
    if "larepublica.pe" in url:
        fin = next((i for i, p in enumerate(partes) if re.fullmatch(r"\d{4}", p)), len(partes))
        seccion = "/".join(partes[:fin])
    else:
        seccion = "/".join(partes[:-1])
    return seccion, NOMBRES.get(MAPA_SECCIONES.get(seccion.split("/")[0]))


@st.cache_data(ttl=1800, show_spinner="Leyendo los titulares de hoy…")
def titulares_de_hoy(por_feed=15):
    """Lee los RSS oficiales: pausa de 1-2 s entre solicitudes, máx. 1 reintento y caché de 30 min."""
    filas = []
    sesion = requests.Session()
    sesion.headers["User-Agent"] = USER_AGENT
    for i, (medio, url) in enumerate(FEEDS):
        if i:
            time.sleep(random.uniform(1, 2))
        r = None
        for _ in range(2):
            try:
                r = sesion.get(url, timeout=20)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                r = None
            time.sleep(random.uniform(1, 2))
        if r is None or r.status_code != 200:
            continue
        for it in BeautifulSoup(r.content, "xml").find_all("item")[:por_feed]:
            enlace = (it.link.get_text() if it.link else "").split("?")[0].strip()
            titular = re.sub(r"\s+", " ", it.title.get_text() if it.title else "").strip()
            if titular and enlace:
                seccion, real = seccion_y_categoria(enlace)
                try:
                    hora = parsedate_to_datetime(it.pubDate.get_text()).astimezone(LIMA).strftime("%d/%m %H:%M")
                except (AttributeError, TypeError, ValueError):
                    hora = ""
                filas.append({"medio": medio, "titular": titular, "url": enlace, "seccion": seccion,
                              "real": real, "hora": hora})
    hoy = pd.DataFrame(filas).drop_duplicates("url").drop_duplicates("titular") if filas else pd.DataFrame()
    if not hoy.empty:
        probs = clasificar(hoy["titular"])
        hoy["pred"] = [max(p, key=p.get) for p in probs]
        hoy["prob"] = [max(p.values()) for p in probs]
    return hoy.reset_index(drop=True), datetime.now(LIMA).strftime("%H:%M")


def estilo(fig, alto=None):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", bargap=0.4,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font_size=13), **({"height": alto} if alto else {}))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def barras_probabilidad(probs, ganadora):
    fig = go.Figure(go.Bar(
        x=[probs[c] for c in ORDEN], y=ORDEN, orientation="h",
        marker_color=[COLOR_CAT[c] if c == ganadora else GRIS for c in ORDEN],
        text=[f"{probs[c]:.0%}" for c in ORDEN], textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 1.15], visible=False),
                      yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]))
    return estilo(fig, 200)


def usar_titular_de_hoy():
    """Callback del botón 🎲: copia un titular real de hoy (de las 4 secciones) al campo de texto."""
    hoy, _ = titulares_de_hoy()
    candidatos = hoy[hoy["real"].notna()] if not hoy.empty else hoy
    if candidatos.empty:
        st.session_state.aviso = "No se pudieron leer los titulares de hoy; escribe uno."
        return
    fila = candidatos.sample(1).iloc[0]
    st.session_state.titular = fila["titular"]
    st.session_state.origen = {"medio": fila["medio"], "real": fila["real"], "url": fila["url"], "titular": fila["titular"]}
    st.session_state.clasificar = True


df = cargar_datos()
metricas = cargar_metricas()
ss = st.session_state
ss.setdefault("titular", "")
ss.setdefault("origen", None)
ss.setdefault("clasificar", False)

# ================================================================ Encabezado
st.title("¿Pueden convertir titulares peruanos en un clasificador de noticias?")
if metricas:
    st.markdown(f"**Sí.** Un DistilBERT entrenado con 400 titulares de *La República* y *El Comercio*, etiquetados "
                f"por un LLM local, clasifica noticias nuevas en 4 categorías con **F1 macro de "
                f"{metricas['test_f1_macro']:.2f}** (frente a {metricas['zero_shot_f1_macro']:.2f} sin entrenar).")
    k1, k2, k3 = st.columns(3)
    k1.metric("Noticias recolectadas", len(df), "200 por medio", delta_color="off", delta_arrow="off")
    k2.metric("F1 macro en test", f"{metricas['test_f1_macro']:.2f}",
              f"{metricas['test_f1_macro'] - metricas['zero_shot_f1_macro']:+.2f} vs. zero-shot",
              help="Evaluado en 60 noticias reservadas que el modelo nunca vio al entrenar.")
    if metricas.get("auditoria_acuerdo_pct") is not None:
        k3.metric("Acuerdo LLM vs. humano", f"{metricas['auditoria_acuerdo_pct']:.0f}%",
                  f"{metricas['auditoria_acuerdos']}/{metricas['auditoria_total']} casos auditados",
                  delta_color="off", delta_arrow="off")

with st.expander("Cómo se hizo"):
    st.markdown("""
1. **Datos:** 400 noticias (opción C: 200 por medio, 50 por categoría) de RSS, portadas y sitemaps, respetando
   `robots.txt` y con pausas de 1–2 s. La etiqueta de referencia es la sección del diario.
2. **Línea base:** zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli`, sin entrenamiento.
3. **Supervisión débil:** `qwen2.5:7b` (Ollama) etiqueta las 400 noticias; una persona audita 20 al azar.
4. **Modelo:** `distilbert-base-multilingual-cased` entrenado con las etiquetas del LLM (split 70/15/15 estratificado,
   `random_state=42`) y evaluado contra la sección del diario en el test reservado.
""")

# ================================================================ Filtros (pestaña Explorar el corpus)
st.sidebar.header("Filtros")
st.sidebar.caption("Se aplican a **Explorar el corpus**.")
medios_sel = st.sidebar.multiselect("Medio", list(MEDIOS.values()), default=list(MEDIOS.values()))
cats_sel = st.sidebar.multiselect("Categoría", ORDEN, default=ORDEN)
filtrado = df[df["medio_nombre"].isin(medios_sel) & df["categoria"].isin(cats_sel)]

tab_clasificar, tab_hoy, tab_modelo, tab_corpus = st.tabs(
    ["Clasificar un titular", "Titulares de hoy", "Resultados del modelo", "Explorar el corpus"])

# ================================================================ 1. Probador en vivo
with tab_clasificar:
    st.text_area("Escribe o pega un titular", key="titular", height=90,
                 placeholder="Ej.: Congreso aprueba nuevo retiro de fondos de AFP")
    b1, b2 = st.columns([1, 2])
    if b1.button("Clasificar", type="primary", width="stretch"):
        ss.clasificar, ss.origen = True, None
    b2.button("🎲 Probar con un titular real de hoy", on_click=usar_titular_de_hoy, width="stretch",
              help="Toma al azar un titular publicado hoy en La República o El Comercio.")
    if ss.pop("aviso", None):
        st.warning("No se pudieron leer los titulares de hoy; escribe uno.")

    if ss.clasificar:
        ss.clasificar = False
        texto = ss.titular.strip()
        if not texto:
            st.warning("Escribe un titular primero.")
        elif not modelo_disponible():
            st.error("No se encontraron los pesos del modelo. Ejecuta `02_clasificacion.ipynb` "
                     "o define `MODELO_CLASIFICADOR` con un repo de Hugging Face.")
        else:
            probs = clasificar([texto])[0]
            mejor = max(probs, key=probs.get)
            st.divider()
            r1, r2 = st.columns([1, 1], vertical_alignment="center")
            r1.metric("Categoría predicha", mejor, f"{probs[mejor]:.0%} de probabilidad", delta_color="off",
                      delta_arrow="off")
            r2.plotly_chart(barras_probabilidad(probs, mejor), width="stretch")
            origen = ss.origen if ss.origen and ss.origen["titular"] == texto else None
            if origen:
                ok = origen["real"] == mejor
                st.markdown(f"{'✅' if ok else '❌'} Publicado hoy por **{origen['medio']}** en la sección "
                            f"**{origen['real']}** · [ver noticia]({origen['url']})")
            if probs[mejor] < 0.6:
                st.info("Confianza baja: el titular puede ser ambiguo o de un tema fuera de las 4 categorías "
                        "(el modelo siempre elige una).")

# ================================================================ 2. Titulares de hoy
with tab_hoy:
    if not modelo_disponible():
        st.info("Esta pestaña necesita el modelo entrenado.")
    else:
        hoy, hora = titulares_de_hoy()
        if hoy.empty:
            st.warning("No se pudieron leer los RSS en este momento.")
        else:
            evaluables = hoy[hoy["real"].notna()]
            st.caption(f"Leídos a las {hora} de los RSS oficiales de ambos medios (se renuevan cada 30 min). "
                       "Ninguno se usó para entrenar; la sección real sale de la URL.")
            h1, h2 = st.columns(2)
            h1.metric("Titulares clasificados", len(evaluables), f"+{len(hoy) - len(evaluables)} de otras secciones",
                      delta_color="off", delta_arrow="off")
            h2.metric("Acierto en noticias nuevas", f"{(evaluables['pred'] == evaluables['real']).mean():.0%}",
                      help="Predicción del modelo vs. sección real de la URL.")
            vista = evaluables.assign(acierto=evaluables["pred"] == evaluables["real"])
            st.dataframe(vista[["titular", "pred", "prob", "real", "acierto", "medio", "url"]],
                         hide_index=True, width="stretch", height=420, column_config={
                             "titular": st.column_config.TextColumn("titular", width="large"),
                             "pred": "predicción",
                             "prob": st.column_config.ProgressColumn("confianza", format="percent", min_value=0,
                                                                     max_value=1),
                             "real": "sección real", "acierto": st.column_config.CheckboxColumn("✓"),
                             "url": st.column_config.LinkColumn("enlace", display_text="abrir")})

# ================================================================ 3. Resultados del modelo
with tab_modelo:
    if not metricas:
        st.info("Ejecuta `02_clasificacion.ipynb` para generar `modelo/metricas.json`.")
    else:
        lat = metricas["latencia_ms"]
        comp = pd.DataFrame([
            {"modelo": "Zero-shot", "F1": metricas["zero_shot_f1_macro"], "ms": lat["zero_shot"]},
            {"modelo": "LLM qwen2.5:7b", "F1": metricas["llm_test_f1_macro"], "ms": lat["llm"]},
            {"modelo": "DistilBERT (elegido)", "F1": metricas["test_f1_macro"], "ms": lat["fine_tuned"]},
        ])
        fig = go.Figure(go.Bar(
            x=comp["F1"], y=comp["modelo"], orientation="h",
            marker_color=[DESTACADO if "elegido" in m else GRIS for m in comp["modelo"]],
            text=[f"{f:.2f} · {ms / 1000:.1f} s/titular" if ms >= 1000 else f"{f:.2f} · {ms:.0f} ms/titular"
                  for f, ms in zip(comp["F1"], comp["ms"])], textposition="outside",
            hovertemplate="%{y}: F1 macro %{x:.3f}<extra></extra>"))
        fig.update_layout(title="F1 macro en el test reservado y tiempo por titular (CPU)",
                          xaxis=dict(range=[0, 1.45], visible=False))
        st.plotly_chart(estilo(fig, 230), width="stretch")
        st.caption(f"DistilBERT queda a {metricas['llm_test_f1_macro'] - metricas['test_f1_macro']:.2f} del LLM, "
                   f"pero es ~{lat['llm'] / lat['fine_tuned']:.0f} veces más rápido y no necesita un servidor aparte.")

        cm = pd.DataFrame(metricas["matriz_confusion"], index=ORDEN, columns=ORDEN)
        pct = cm.div(cm.sum(axis=1), axis=0)
        fig = go.Figure(go.Heatmap(z=pct.values, x=ORDEN, y=ORDEN, colorscale=[[0, "#eef4fc"], [1, "#1c5cab"]],
                                   zmin=0, zmax=1, showscale=False, xgap=2, ygap=2, customdata=cm.values,
                                   hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{customdata} noticias<extra></extra>"))
        for i, real in enumerate(ORDEN):
            for j, pred in enumerate(ORDEN):
                if cm.iloc[i, j]:
                    fig.add_annotation(x=pred, y=real, text=f"{cm.iloc[i, j]}", showarrow=False,
                                       font=dict(color="white" if pct.iloc[i, j] > 0.5 else "#1f2328", size=14))
        fig.update_layout(title="Matriz de confusión en test", xaxis_title="predicción", yaxis_title="sección real",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(estilo(fig, 360), width="stretch")
        st.caption("La confusión principal es **economía → política**: la economía peruana se narra a través de "
                   "ministros, Petroperú o el Gobierno, y el modelo heredó ese sesgo del LLM que lo etiquetó.")

# ================================================================ 4. Explorar el corpus
with tab_corpus:
    st.caption(f"{len(filtrado)} de {len(df)} noticias según los filtros de la barra lateral.")
    if not filtrado.empty:
        dist = filtrado.groupby(["categoria", "categoria_llm"]).size().rename("noticias").reset_index()
        dist["%"] = dist["noticias"] / dist.groupby("categoria")["noticias"].transform("sum")
        fig = px.bar(dist, x="%", y="categoria", color="categoria_llm", orientation="h",
                     color_discrete_map=COLOR_CAT, custom_data=["noticias", "categoria_llm"],
                     category_orders={"categoria": [c for c in ORDEN if c in dist["categoria"].unique()],
                                      "categoria_llm": ORDEN},
                     labels={"categoria": "", "categoria_llm": "Etiqueta del LLM", "%": ""})
        fig.update_traces(marker_line_color="white", marker_line_width=1.5,
                          hovertemplate="Sección %{y}<br>LLM: %{customdata[1]}<br>%{customdata[0]} noticias (%{x:.0%})<extra></extra>")
        fig.update_layout(title="¿Cómo etiquetó el LLM cada sección del diario?",
                          xaxis=dict(tickformat=".0%"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(estilo(fig, 300), width="stretch")

    st.dataframe(filtrado[["titular", "medio_nombre", "categoria", "categoria_llm", "fecha", "url"]],
                 hide_index=True, width="stretch", height=360, column_config={
                     "titular": st.column_config.TextColumn("titular", width="large"),
                     "medio_nombre": "medio", "categoria": "sección", "categoria_llm": "LLM",
                     "fecha": st.column_config.DateColumn("fecha"),
                     "url": st.column_config.LinkColumn("enlace", display_text="abrir")})
    st.download_button("Descargar CSV", filtrado[["titular", "bajada", "medio", "seccion_original", "url", "fecha",
                                                  "etiqueta", "etiqueta_llm", "confianza"]].to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

st.caption("Tarea 1 · Diplomatura Python Intermedio · Datos de larepublica.pe y elcomercio.pe con fines académicos.")
