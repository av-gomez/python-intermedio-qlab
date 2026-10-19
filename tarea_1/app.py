"""Dashboard Streamlit – Tarea 1: Noticias peruanas, del titular al clasificador.

"La Portada Clasificada": la app clasifica en vivo los titulares del día, propone un juego contra el modelo
y permite escribir un titular propio.

Ejecutar desde la carpeta tarea_1/:
    streamlit run app.py
"""
import html
import json
import math
import os
import random
import re
import time
from collections import Counter
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
COLOR_MEDIO = {"La República": "#7b6ce0", "El Comercio": "#8a8984"}
GRIS, DESTACADO, TINTA, PAPEL = "#9c9b95", "#2a78d6", "#1f1d1a", "#fbf9f4"

# Mismo mapeo sección → categoría que 01_recoleccion.ipynb
MAPA_SECCIONES = {"politica": "politica", "economia": "economia", "deportes": "deportes",
                  "deporte-total": "deportes", "ciencia": "ciencia_tecnologia", "tecnologia": "ciencia_tecnologia"}
# RSS oficiales (permitidos en robots.txt): portada + las 4 secciones de cada medio
_EC = "https://elcomercio.pe/arc/outboundfeeds/rss"
FEEDS = [("La República", "https://larepublica.pe/rss/home.xml")] +         [("La República", f"https://larepublica.pe/rss/{s}.xml") for s in ["politica", "economia", "deportes", "ciencia"]] +         [("El Comercio", f"{_EC}/?outputType=xml")] +         [("El Comercio", f"{_EC}/category/{s}/?outputType=xml") for s in ["politica", "economia", "deporte-total", "tecnologia"]]
USER_AGENT = "TareaNLP-Diplomatura-PythonIntermedio/1.0 (dashboard academico; lectura de RSS cada 30 min, 1-2 s entre solicitudes)"

STOPWORDS = set("""
a al algo algunos ante antes aquí así aun aún cada casi como con contra cual cuales cuando cuándo cómo de del desde
donde dónde durante e el él ella ellas ellos en entre era es esa ese eso esta este esto estos estas está están fue fueron
ha han hasta hay la las le les lo los más mas me mi muy no nos o otra otro otros para pero por porque qué que quien
quién se sea según ser si sí sin sobre su sus también tan te tiene tienen todo todos tras tu un una uno unos unas va van
ya yo y hoy será son sido tras cuál vs ver dónde así cómo nuevo nueva nuevos qué cuánto cuánta tras dice afirma
según dijo sus ser este esta todo puede podría años año día días hora horas tras sobre mientras video vivo envivo
hace bajo enero febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre
lunes martes miércoles jueves viernes sábado domingo
""".split())

st.set_page_config(page_title="La Portada Clasificada", page_icon="🗞️", layout="wide")

# ================================================================ Estilo de periódico
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
h1, h2, h3, .cabecera, .titular-card a { font-family: 'Playfair Display', Georgia, serif !important; }
.block-container { padding-top: 3.6rem; max-width: 1280px; }
.cabecera { text-align: center; font-size: 3.4rem !important; font-weight: 800; letter-spacing: -1px; line-height: 1.05 !important;
            margin: 0 !important; color: #1f1d1a; }
.fechero { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; border-top: 3px double #1f1d1a;
           border-bottom: 1px solid #1f1d1a; padding: 4px 2px; margin: 10px 0 18px; font-size: .78rem;
           letter-spacing: .08em; text-transform: uppercase; color: #52514e; }
.pregunta { text-align: center; font-family: 'Source Serif 4', Georgia, serif; font-size: 1.55rem !important; font-weight: 600;
            margin: 0 0 4px; color: #1f1d1a; }
.respuesta { text-align: center; font-family: 'Source Serif 4', Georgia, serif; font-size: 1.05rem !important; color: #3d3c39;
             max-width: 860px; margin: 0 auto 18px; }
.seccion-cab { font-family: 'Playfair Display', Georgia, serif; font-weight: 800; font-size: 1.02rem; padding: 4px 0 6px; white-space: nowrap; gap: 6px;
               border-bottom: 3px solid #1f1d1a; margin-bottom: 8px; display: flex; justify-content: space-between; }
.seccion-cab span { font-family: sans-serif; font-weight: 400; font-size: .78rem; color: #52514e; align-self: end; }
.titular-card { border-left: 4px solid var(--c); background: #fffdf8; padding: 8px 10px 7px; margin-bottom: 9px;
                border-radius: 0 4px 4px 0; box-shadow: 0 1px 0 rgba(0,0,0,.06); }
.titular-card a { color: #1f1d1a !important; text-decoration: none; font-size: .95rem; font-weight: 600; line-height: 1.28; }
.titular-card a:hover { text-decoration: underline; }
.meta { font-size: .74rem; color: #52514e; margin-top: 4px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.barra { height: 4px; background: #e6e2d8; border-radius: 2px; margin-top: 5px; }
.barra > div { height: 4px; border-radius: 2px; background: var(--c); }
.tag { border: 1px solid #c9c5ba; border-radius: 3px; padding: 0 5px; }
.tag-error { border-color: #b3261e; color: #b3261e; }
.tag-fuera { border-style: dashed; }
.juego { text-align: center; background: #fffdf8; border: 1px solid #ddd8cc; border-top: 4px solid #1f1d1a;
         padding: 26px 30px; margin: 6px auto 14px; max-width: 900px; }
.juego .q { font-family: 'Playfair Display', Georgia, serif; font-size: 1.9rem; font-weight: 600; line-height: 1.2; color: #1f1d1a; }
.juego .m { font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: #52514e; margin-bottom: 10px; }
div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #e3dfd4; border-radius: 4px; padding: 10px 14px; }
@media (max-width: 640px) { .cabecera { font-size: 2.2rem !important; } .juego .q { font-size: 1.35rem; } }
</style>
""", unsafe_allow_html=True)


# ================================================================ Datos y modelo
@st.cache_data
def cargar_datos():
    noticias = pd.read_csv(BASE / "datos" / "noticias.csv")
    llm = pd.read_csv(BASE / "datos" / "etiquetas_llm.csv")[["url", "etiqueta_llm", "confianza"]]
    split = pd.read_csv(BASE / "datos" / "split.csv")[["url", "particion"]]
    df = noticias.merge(llm, on="url", how="left").merge(split, on="url", how="left")
    df["categoria"] = df["etiqueta"].map(NOMBRES)
    df["categoria_llm"] = df["etiqueta_llm"].map(NOMBRES)
    df["medio_nombre"] = df["medio"].map(MEDIOS)
    df["coincide_llm"] = df["etiqueta"] == df["etiqueta_llm"]
    df["largo"] = df["titular"].str.len()
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


@st.cache_data(show_spinner="Clasificando el conjunto de test…")
def predicciones_test(titulares):
    return clasificar(titulares)


def seccion_y_categoria(url):
    """Sección según la URL (mismas reglas que 01_recoleccion.ipynb) y su categoría, o None si no es de las 4."""
    partes = [p for p in urlparse(url).path.split("/") if p]
    if "larepublica.pe" in url:
        fin = next((i for i, p in enumerate(partes) if re.fullmatch(r"\d{4}", p)), len(partes))
        seccion = "/".join(partes[:fin])
    else:
        seccion = "/".join(partes[:-1])
    return seccion, NOMBRES.get(MAPA_SECCIONES.get(seccion.split("/")[0]))


@st.cache_data(ttl=1800, show_spinner="Leyendo los RSS de hoy y clasificando los titulares…")
def portada_en_vivo(por_feed=15):
    """Lee los RSS oficiales (permitidos en robots.txt): pausa de 1-2 s, máx. 1 reintento y caché de 30 min."""
    filas, errores = [], []
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
            errores.append(url)
            continue
        for it in BeautifulSoup(r.content, "xml").find_all("item")[:por_feed]:
            enlace = (it.link.get_text() if it.link else "").split("?")[0].strip()
            titular = re.sub(r"\s+", " ", it.title.get_text() if it.title else "").strip()
            if not titular or not enlace:
                continue
            seccion, real = seccion_y_categoria(enlace)
            try:
                hora = parsedate_to_datetime(it.pubDate.get_text()).astimezone(LIMA).strftime("%d/%m %H:%M")
            except (AttributeError, TypeError, ValueError):
                hora = ""
            filas.append({"medio": medio, "titular": titular, "url": enlace, "seccion": seccion, "real": real, "hora": hora})
    vivo = pd.DataFrame(filas).drop_duplicates("url").drop_duplicates("titular") if filas else pd.DataFrame()
    if not vivo.empty:
        probs = clasificar(vivo["titular"])
        vivo["pred"] = [max(p, key=p.get) for p in probs]
        vivo["prob"] = [max(p.values()) for p in probs]
    return vivo, errores, datetime.now(LIMA).strftime("%H:%M")


@st.cache_resource
def indice_similitud(titulares):
    vectorizador = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True,
                                   stop_words=sorted(STOPWORDS))
    return vectorizador, vectorizador.fit_transform(titulares)


def tokens(texto):
    return [t for t in re.findall(r"[a-záéíóúñü]+", texto.lower()) if len(t) > 2 and t not in STOPWORDS]


def palabras_distintivas(datos, n=8, minimo=3):
    """Log-odds suavizado de cada palabra en una categoría frente al resto de las categorías."""
    conteos = {c: Counter(t for tit in g["titular"] for t in set(tokens(tit))) for c, g in datos.groupby("categoria")}
    total = Counter()
    for c in conteos.values():
        total.update(c)
    vocab = len(total)
    filas = []
    for cat, cnt in conteos.items():
        n_cat, n_resto = sum(cnt.values()), sum(total.values()) - sum(cnt.values())
        for palabra, k in cnt.items():
            if k < minimo:
                continue
            resto = total[palabra] - k
            score = math.log((k + 1) / (n_cat + vocab)) - math.log((resto + 1) / (n_resto + vocab))
            filas.append({"categoria": cat, "palabra": palabra, "titulares": k, "score": score})
    if not filas:
        return pd.DataFrame(columns=["categoria", "palabra", "titulares", "score"])
    top = pd.DataFrame(filas).sort_values("score", ascending=False).groupby("categoria").head(n)
    return top.sort_values(["categoria", "titulares"], ascending=[True, False])


def estilo(fig, alto=None):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", bargap=0.35,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=TINTA,
                      hoverlabel=dict(font_size=13), **({"height": alto} if alto else {}))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def tarjeta(titular, url, color, meta_html, prob=None):
    barra = f'<div class="barra"><div style="width:{prob:.0%}"></div></div>' if prob is not None else ""
    return (f'<div class="titular-card" style="--c:{color}"><a href="{html.escape(url)}" target="_blank">'
            f'{html.escape(titular)}</a><div class="meta">{meta_html}</div>{barra}</div>')


def barras_probabilidad(probs, ganadora):
    fig = go.Figure(go.Bar(
        x=[probs[c] for c in ORDEN], y=ORDEN, orientation="h",
        marker_color=[COLOR_CAT[c] if c == ganadora else GRIS for c in ORDEN],
        text=[f"{probs[c]:.0%}" for c in ORDEN], textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(title="Probabilidad por categoría", xaxis=dict(range=[0, 1.12], tickformat=".0%"),
                      yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]))
    return estilo(fig, 250)


df = cargar_datos()
metricas = cargar_metricas()
hoy = datetime.now(LIMA)
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]

# ================================================================ Cabecera
st.markdown('<p class="cabecera">La Portada Clasificada</p>', unsafe_allow_html=True)
st.markdown(f'<div class="fechero"><span>Lima, {DIAS[hoy.weekday()]} {hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}</span>'
            f'<span>Tarea 1 · Diplomatura Python Intermedio</span><span>La República × El Comercio</span></div>',
            unsafe_allow_html=True)
st.markdown('<p class="pregunta">¿Pueden convertir titulares peruanos en un clasificador de noticias?</p>',
            unsafe_allow_html=True)
if metricas:
    st.markdown(
        f'<p class="respuesta"><b>Sí.</b> Con {len(df)} titulares etiquetados por un LLM local, un DistilBERT ajustado '
        f'clasifica noticias nuevas en política, economía, deportes y ciencia con <b>F1 macro de '
        f'{metricas["test_f1_macro"]:.2f}</b> en un test reservado, frente a {metricas["zero_shot_f1_macro"]:.2f} '
        f'de un modelo sin entrenar. Compruébalo abajo: no necesitas escribir nada.</p>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Noticias recolectadas", f"{len(df)}", "200 por medio · 50 por categoría", delta_color="off", delta_arrow="off",
              help="Descargadas el 18/10/2026 de RSS, portadas de sección y sitemaps, respetando robots.txt.")
    k2.metric("F1 macro en test", f"{metricas['test_f1_macro']:.2f}",
              f"{metricas['test_f1_macro'] - metricas['zero_shot_f1_macro']:+.2f} vs. sin entrenar",
              help="DistilBERT multilingüe ajustado, evaluado en 60 noticias que nunca vio al entrenar.")
    if metricas.get("auditoria_acuerdo_pct") is not None:
        k3.metric("Acuerdo LLM vs. humano", f"{metricas['auditoria_acuerdo_pct']:.0f}%",
                  f"{metricas['auditoria_acuerdos']} de {metricas['auditoria_total']} casos", delta_color="off", delta_arrow="off",
                  help="Concordancia entre el LLM y una auditoría humana de 20 etiquetas elegidas al azar.")
    k4.metric("Tiempo por titular", f"~{metricas['latencia_ms']['fine_tuned']} ms",
              f"el LLM tarda ~{metricas['latencia_ms']['llm'] / 1000:.0f} s", delta_color="off", delta_arrow="off",
              help="Medido en CPU en 02_clasificacion.ipynb.")

with st.expander("Cómo se hizo (resumen metodológico)"):
    st.markdown("""
1. **Recolección** – 400 noticias (opción C, comparada): RSS oficiales, portadas de sección y sitemaps, con `robots.txt`
   respetado y pausas de 1–2 s. La **etiqueta de referencia** es la sección del diario mapeada a 4 categorías; las
   noticias de otras secciones se descartaron.
2. **Línea base** – modelo zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli`, sin entrenamiento.
3. **Supervisión débil** – `qwen2.5:7b` (Ollama, local) etiqueta las 400 noticias; una persona audita 20 al azar.
4. **Fine-tuning** – `distilbert-base-multilingual-cased` entrenado con las etiquetas del LLM (split estratificado
   70/15/15, `random_state=42`) y evaluado contra la sección del diario.
5. **Esta portada** – lee los RSS oficiales de hoy (10 solicitudes con pausas de 1–2 s, caché de 30 min) y los clasifica con el modelo en tiempo real.
""")

# ================================================================ Filtros
st.sidebar.markdown("### Filtros del corpus")
st.sidebar.caption("Se aplican a las pestañas **Explorar el corpus** y **Datos**.")
medios_sel = st.sidebar.multiselect("Medio", list(MEDIOS.values()), default=list(MEDIOS.values()))
cats_sel = st.sidebar.multiselect("Categoría (sección del diario)", ORDEN, default=ORDEN)
busqueda = st.sidebar.text_input("Buscar en titulares", placeholder="ej.: Keiko, dólar, Universitario")
solo_desacuerdo = st.sidebar.toggle("Solo noticias donde el LLM no coincide con la sección")

filtrado = df[df["medio_nombre"].isin(medios_sel) & df["categoria"].isin(cats_sel)]
if busqueda.strip():
    filtrado = filtrado[filtrado["titular"].str.contains(busqueda.strip(), case=False, regex=False)]
if solo_desacuerdo:
    filtrado = filtrado[~filtrado["coincide_llm"]]
st.sidebar.metric("Noticias seleccionadas", f"{len(filtrado)} de {len(df)}")

tab_vivo, tab_juego, tab_escribir, tab_modelo, tab_corpus, tab_datos = st.tabs(
    ["📰 Portada en vivo", "🎯 ¿Le ganas a la IA?", "✍️ Escribe tu titular", "📈 Resultados del modelo",
     "🔍 Explorar el corpus", "📄 Datos"])

# ================================================================ 1. Portada en vivo
with tab_vivo:
    if not modelo_disponible():
        st.error("No se encontraron los pesos del modelo. Ejecuta `02_clasificacion.ipynb` o define "
                 "`MODELO_CLASIFICADOR` con un repo de Hugging Face.")
    else:
        vivo, errores, hora_lectura = portada_en_vivo()
        if vivo.empty:
            st.warning("No se pudieron leer los RSS en este momento. Prueba el juego o escribe un titular.")
        else:
            evaluables = vivo[vivo["real"].notna()]
            fuera = vivo[vivo["real"].isna()]
            st.caption(f"Titulares leídos a las {hora_lectura} de {len(FEEDS) - len(errores)} RSS oficiales de La República "
                       "y El Comercio (portada y secciones; se renuevan cada 30 min). Ninguno se usó para entrenar; "
                       "la sección real se obtiene de la URL de cada noticia.")
            v1, v2, v3 = st.columns(3)
            v1.metric("Titulares de hoy clasificados", len(vivo))
            if len(evaluables):
                v2.metric("Acierto en vivo", f"{(evaluables['pred'] == evaluables['real']).mean():.0%}",
                          f"en {len(evaluables)} noticias de las 4 secciones", delta_color="off", delta_arrow="off",
                          help="Predicción vs. sección real de la URL (solo noticias de las 4 categorías).")
            v3.metric("De otras secciones", len(fuera), "sociedad, mundo, espectáculos…", delta_color="off", delta_arrow="off",
                      help="El modelo siempre elige una de las 4 categorías: estas noticias muestran sus límites.")

            ver_fuera = st.toggle("Incluir noticias de otras secciones (fuera de las 4 categorías)", value=False)
            mostrar = vivo if ver_fuera else evaluables
            for col, cat in zip(st.columns(4, gap="medium"), ORDEN):
                grupo = mostrar[mostrar["pred"] == cat].sort_values("prob", ascending=False)
                col.markdown(f'<div class="seccion-cab" style="border-color:{COLOR_CAT[cat]}">{cat}'
                             f'<span>{len(grupo)} titulares</span></div>', unsafe_allow_html=True)
                bloques = []
                for _, n in grupo.head(12).iterrows():
                    meta = f'<span class="tag">{n["medio"]}</span><span>{n["hora"]}</span><span>{n["prob"]:.0%}</span>'
                    if pd.isna(n["real"]):
                        meta += f'<span class="tag tag-fuera">sección: {html.escape(n["seccion"].split("/")[0] or "portada")}</span>'
                    elif n["real"] != cat:
                        meta += f'<span class="tag tag-error">✗ era {n["real"]}</span>'
                    bloques.append(tarjeta(n["titular"], n["url"], COLOR_CAT[cat], meta, n["prob"]))
                col.markdown("".join(bloques) or "<i>Sin titulares</i>", unsafe_allow_html=True)
            st.caption("La barra de cada titular es la probabilidad de la categoría elegida. **✗ era …** marca un error "
                       "del modelo; las etiquetas punteadas son noticias de otras secciones.")

# ================================================================ 2. Juego
with tab_juego:
    test = df[df["particion"] == "test"].reset_index(drop=True)
    if not modelo_disponible() or test.empty:
        st.info("El juego necesita el modelo entrenado y datos/split.csv.")
    else:
        preds = predicciones_test(tuple(test["titular"]))
        ss = st.session_state
        if "orden_juego" not in ss:
            ss.orden_juego = random.sample(range(len(test)), len(test))
            ss.ronda, ss.puntos_tu, ss.puntos_ia, ss.respuesta = 0, 0, 0, None
        i = ss.orden_juego[ss.ronda % len(test)]
        noticia, probs = test.iloc[i], preds[i]
        pred_ia = max(probs, key=probs.get)

        st.markdown(f'<div class="juego"><div class="m">{noticia["medio_nombre"]} · {noticia["fecha"]:%d/%m/%Y} · '
                    f'titular {ss.ronda + 1}</div><div class="q">{html.escape(noticia["titular"])}</div></div>',
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>¿En qué sección lo publicó el diario?</p>", unsafe_allow_html=True)
        for b, cat in zip(st.columns([1, 2, 2, 2, 2, 1])[1:5], ORDEN):
            if b.button(cat, key=f"j_{cat}", width="stretch", disabled=ss.respuesta is not None):
                ss.respuesta = cat
                ss.puntos_tu += cat == noticia["categoria"]
                ss.puntos_ia += pred_ia == noticia["categoria"]
                st.rerun()

        if ss.respuesta is not None:
            real = noticia["categoria"]
            r1, r2, r3 = st.columns(3)
            r1.metric("Tu respuesta", ss.respuesta, "✓ correcta" if ss.respuesta == real else "✗ incorrecta",
                      delta_color="normal" if ss.respuesta == real else "inverse", delta_arrow="off")
            r2.metric("La IA dijo", pred_ia, f"{'✓' if pred_ia == real else '✗'} con {probs[pred_ia]:.0%} de probabilidad",
                      delta_color="normal" if pred_ia == real else "inverse", delta_arrow="off")
            r3.metric("Sección real del diario", real, f"el LLM maestro dijo {noticia['categoria_llm']}", delta_color="off", delta_arrow="off")
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(barras_probabilidad(probs, pred_ia), width="stretch")
            with c2:
                st.markdown(f"**Bajada:** {noticia['bajada']}")
                st.markdown(f"[Leer la noticia original ↗]({noticia['url']})")
                if st.button("Siguiente titular →", type="primary"):
                    ss.ronda += 1
                    ss.respuesta = None
                    st.rerun()

        jugadas = ss.ronda + (ss.respuesta is not None)
        if jugadas:
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Tus aciertos", f"{ss.puntos_tu} / {jugadas}")
            m2.metric("Aciertos de la IA", f"{ss.puntos_ia} / {jugadas}")
            m3.metric("Marcador", "¡Vas ganando!" if ss.puntos_tu > ss.puntos_ia else
                      ("Empate" if ss.puntos_tu == ss.puntos_ia else "Gana la IA"))
        st.caption(f"Los {len(test)} titulares del juego pertenecen al **test reservado**: el modelo nunca los vio al entrenar.")

# ================================================================ 3. Escribe tu titular (probador en vivo)
with tab_escribir:
    st.markdown("### Escribe o pega un titular nuevo")
    ejemplos = ["Congreso aprueba nuevo retiro de fondos de AFP",
                "Universitario vence a Alianza Lima en el clásico y se acerca al título",
                "NASA confirma hallazgo de agua en un asteroide cercano a la Tierra",
                "BCR mantiene su tasa de referencia por tercer mes consecutivo",
                "Shakira anuncia un nuevo concierto en Lima"]
    elegido = st.pills("Ideas para probar (la última no pertenece a ninguna de las 4 categorías)", ejemplos,
                       selection_mode="single")
    with st.form("probador"):
        titular = st.text_area("Titular", value=elegido or "", height=80,
                               placeholder="Pega aquí un titular de cualquier diario peruano…")
        enviado = st.form_submit_button("Clasificar", type="primary")

    if enviado:
        if not titular.strip():
            st.warning("Escribe un titular primero.")
        elif not modelo_disponible():
            st.error("No se encontraron los pesos del modelo en `modelo/`.")
        else:
            probs = clasificar([titular.strip()])[0]
            mejor = max(probs, key=probs.get)
            segunda = sorted(probs, key=probs.get)[-2]
            r1, r2 = st.columns([2, 3], gap="large")
            with r1:
                st.metric("Categoría predicha", mejor, f"{probs[mejor]:.0%} de probabilidad", delta_color="off", delta_arrow="off")
                if probs[mejor] < 0.6:
                    st.warning(f"**Confianza baja.** Duda con *{segunda}* ({probs[segunda]:.0%}). Puede ser un titular "
                               "ambiguo o de un tema fuera de las 4 categorías (el modelo siempre elige una).")
                else:
                    st.success(f"Confianza alta. Segunda opción: {segunda} ({probs[segunda]:.0%}).")
            r2.plotly_chart(barras_probabilidad(probs, mejor), width="stretch")

            vectorizador, matriz = indice_similitud(tuple(df["titular"]))
            sim = cosine_similarity(vectorizador.transform([titular]), matriz).ravel()
            vecinos = df.assign(similitud=sim).nlargest(5, "similitud")
            vecinos = vecinos[vecinos["similitud"] > 0]
            st.markdown("**Titulares más parecidos del corpus** (similitud de vocabulario, TF-IDF)")
            if vecinos.empty:
                st.caption("Ningún titular del corpus comparte vocabulario con el texto ingresado.")
            else:
                st.markdown("".join(
                    tarjeta(v["titular"], v["url"], COLOR_CAT[v["categoria"]],
                            f'<span class="tag">{v["medio_nombre"]}</span><span>{v["categoria"]}</span>'
                            f'<span>similitud {v["similitud"]:.2f}</span>')
                    for _, v in vecinos.iterrows()), unsafe_allow_html=True)

# ================================================================ 4. Resultados del modelo
with tab_modelo:
    if not metricas:
        st.info("Ejecuta `02_clasificacion.ipynb` para generar `modelo/metricas.json`.")
    else:
        st.markdown("### Tres formas de clasificar, comparadas en el test reservado")
        lat = metricas["latencia_ms"]
        comp = pd.DataFrame([
            {"modelo": "Zero-shot (sin entrenar)", "F1 macro": metricas["zero_shot_f1_macro"], "ms": lat["zero_shot"],
             "nota": "40 titulares de test · ~440 MB"},
            {"modelo": "LLM qwen2.5:7b", "F1 macro": metricas["llm_test_f1_macro"], "ms": lat["llm"],
             "nota": "60 titulares de test · 4.7 GB + servidor Ollama"},
            {"modelo": "DistilBERT ajustado ★", "F1 macro": metricas["test_f1_macro"], "ms": lat["fine_tuned"],
             "nota": "60 titulares de test · 541 MB · modelo elegido"},
        ])
        colores = [DESTACADO if "★" in m else GRIS for m in comp["modelo"]]
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = go.Figure(go.Bar(x=comp["F1 macro"], y=comp["modelo"], orientation="h", marker_color=colores,
                                   text=comp["F1 macro"].map("{:.2f}".format), textposition="outside",
                                   customdata=comp["nota"],
                                   hovertemplate="%{y}<br>F1 macro %{x:.3f}<br>%{customdata}<extra></extra>"))
            fig.update_layout(title="Calidad (F1 macro, más es mejor)", xaxis=dict(range=[0, 1.1]))
            st.plotly_chart(estilo(fig, 260), width="stretch")
        with c2:
            fig = go.Figure(go.Bar(x=comp["ms"], y=comp["modelo"], orientation="h", marker_color=colores,
                                   text=[f"{v:,.0f} ms" for v in comp["ms"]], textposition="outside",
                                   hovertemplate="%{y}<br>%{x:,.0f} ms por titular<extra></extra>"))
            fig.update_layout(title="Tiempo por titular en CPU (escala log., menos es mejor)",
                              xaxis=dict(type="log", range=[1, math.log10(lat["llm"]) + 0.6],
                                         tickvals=[10, 100, 1000, 10000], ticktext=["10 ms", "100 ms", "1 s", "10 s"]))
            st.plotly_chart(estilo(fig, 260), width="stretch")
        st.caption(f"El LLM es el más preciso, pero DistilBERT logra casi lo mismo siendo "
                   f"~{lat['llm'] / lat['fine_tuned']:.0f} veces más rápido y sin servidor externo: por eso es el modelo "
                   "de esta portada. El LLM queda como anotador para re-entrenar y para casos dudosos.")

        st.divider()
        c3, c4 = st.columns([3, 2], gap="large")
        with c3:
            cm = pd.DataFrame(metricas["matriz_confusion"], index=ORDEN, columns=ORDEN)
            pct = cm.div(cm.sum(axis=1), axis=0)
            fig = go.Figure(go.Heatmap(z=pct.values, x=ORDEN, y=ORDEN, colorscale=[[0, "#e3eefc"], [1, "#1c5cab"]],
                                       zmin=0, zmax=1, showscale=False, xgap=2, ygap=2, customdata=cm.values,
                                       hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{customdata} noticias (%{z:.0%})<extra></extra>"))
            for i, real in enumerate(ORDEN):  # texto oscuro en celdas claras y blanco en celdas oscuras
                for j, pred in enumerate(ORDEN):
                    if cm.iloc[i, j]:
                        fig.add_annotation(x=pred, y=real, showarrow=False,
                                           text=f"<b>{pct.iloc[i, j]:.0%}</b><br>{cm.iloc[i, j]} noticia{'s' if cm.iloc[i, j] > 1 else ''}",
                                           font=dict(color="white" if pct.iloc[i, j] > 0.5 else "#0b0b0b"))
            fig.update_layout(title="¿En qué se equivoca? Matriz de confusión en test (% por fila)",
                              xaxis_title="Predicción del modelo", yaxis_title="Sección real del diario",
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(estilo(fig, 380), width="stretch")
        with c4:
            f1 = pd.Series(metricas["test_f1_por_clase"]).rename(index=NOMBRES).loc[ORDEN]
            fig = go.Figure(go.Bar(x=f1.values, y=f1.index, orientation="h", marker_color=[COLOR_CAT[c] for c in f1.index],
                                   text=[f"{v:.2f}" for v in f1.values], textposition="outside",
                                   hovertemplate="%{y}: F1 %{x:.3f}<extra></extra>"))
            fig.update_layout(title="F1 por categoría", xaxis=dict(range=[0, 1.12]),
                              yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]))
            st.plotly_chart(estilo(fig, 260), width="stretch")
            st.markdown("**Lectura:** deportes y ciencia tienen un vocabulario propio (clubes, marcadores, IA, "
                        "científicos). La frontera difícil es **economía ↔ política**: la economía peruana se cuenta "
                        "a través de ministros, Petroperú o el Gobierno, y el modelo heredó esa confusión del LLM.")

# ================================================================ 5. Explorar el corpus
with tab_corpus:
    if filtrado.empty:
        st.info("No hay noticias con los filtros actuales.")
    else:
        st.markdown("### ¿El LLM etiqueta igual que el diario?")
        dist = filtrado.groupby(["categoria", "categoria_llm"]).size().rename("noticias").reset_index()
        dist["%"] = dist["noticias"] / dist.groupby("categoria")["noticias"].transform("sum")
        acuerdo = filtrado.groupby("categoria")["coincide_llm"].mean()
        fig = px.bar(dist, x="%", y="categoria", color="categoria_llm", orientation="h",
                     color_discrete_map=COLOR_CAT, custom_data=["noticias", "categoria_llm"],
                     category_orders={"categoria": [c for c in ORDEN if c in acuerdo.index], "categoria_llm": ORDEN},
                     labels={"categoria": "Sección del diario", "categoria_llm": "Etiqueta del LLM", "%": ""})
        fig.update_traces(marker_line_color=PAPEL, marker_line_width=1.5,
                          hovertemplate="Sección: %{y}<br>LLM dijo: %{customdata[1]}<br>%{customdata[0]} noticias (%{x:.0%})<extra></extra>")
        for cat, valor in acuerdo.items():
            fig.add_annotation(x=1.01, y=cat, text=f"{valor:.0%} coincide", showarrow=False, xanchor="left")
        fig.update_layout(title="Distribución de las etiquetas del LLM dentro de cada sección (noticias filtradas)",
                          xaxis=dict(tickformat=".0%", range=[0, 1.18]), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(estilo(fig, 320), width="stretch")
        st.caption("Cada barra es una sección del diario; los colores muestran cómo la etiquetó el LLM. Activa "
                   "*Solo noticias donde el LLM no coincide* en la barra lateral para leer esos casos en **Datos**.")

        st.divider()
        st.markdown("### Palabras que distinguen cada categoría")
        palabras = palabras_distintivas(filtrado)
        if palabras.empty:
            st.caption("Hay muy pocas noticias seleccionadas para calcular palabras distintivas.")
        else:
            cats_presentes = [c for c in ORDEN if c in palabras["categoria"].unique()]
            for col, cat in zip(st.columns(len(cats_presentes)), cats_presentes):
                p = palabras[palabras["categoria"] == cat].sort_values("titulares")
                fig = go.Figure(go.Bar(x=p["titulares"], y=p["palabra"], orientation="h", marker_color=COLOR_CAT[cat],
                                       hovertemplate="«%{y}» aparece en %{x} titulares<extra></extra>"))
                fig.update_layout(title=cat, xaxis_title="titulares")
                col.plotly_chart(estilo(fig, 320), width="stretch")
            st.caption("Palabras más características de cada categoría frente a las demás (log-odds suavizado; solo "
                       "palabras en 3 o más titulares). Filtra por medio para comparar el vocabulario de cada diario.")

        st.divider()
        st.markdown("### ¿Titulan distinto La República y El Comercio?")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            med = filtrado.groupby(["categoria", "medio_nombre"])["largo"].median().reset_index()
            fig = go.Figure()
            for cat in [c for c in ORDEN if c in med["categoria"].unique()]:
                fila = med[med["categoria"] == cat]
                fig.add_trace(go.Scatter(x=fila["largo"], y=[cat] * len(fila), mode="lines",
                                         line=dict(color=GRIS, width=2), showlegend=False, hoverinfo="skip"))
            for medio, color in COLOR_MEDIO.items():
                fila = med[med["medio_nombre"] == medio]
                fig.add_trace(go.Scatter(x=fila["largo"], y=fila["categoria"], mode="markers", name=medio,
                                         marker=dict(size=13, color=color, line=dict(color=PAPEL, width=2)),
                                         hovertemplate=f"{medio}<br>%{{y}}: mediana %{{x:.0f}} caracteres<extra></extra>"))
            fig.update_layout(title="Longitud mediana del titular (caracteres)",
                              yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]), legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(estilo(fig, 300), width="stretch")
        with c2:
            rasgos = filtrado.assign(
                con_cita=filtrado["titular"].str.contains(r"[“\"«]"),
                pregunta=filtrado["titular"].str.contains(r"[¿?]"),
                exclamacion=filtrado["titular"].str.contains(r"[¡!]"),
                en_vivo=filtrado["titular"].str.contains(r"EN VIVO|VIDEO|HOY", case=True),
            ).groupby("medio_nombre")[["largo", "con_cita", "pregunta", "exclamacion", "en_vivo"]].agg(
                {"largo": "median", "con_cita": "mean", "pregunta": "mean", "exclamacion": "mean", "en_vivo": "mean"})
            rasgos.index.name = "medio"
            st.markdown("**Rasgos de estilo** (noticias filtradas)")
            st.dataframe(rasgos, width="stretch", column_config={
                "largo": st.column_config.NumberColumn("largo", format="%d car.", help="Longitud mediana del titular"),
                "con_cita": st.column_config.NumberColumn("citas", format="percent", help="Titulares con comillas"),
                "pregunta": st.column_config.NumberColumn("¿?", format="percent", help="Titulares con pregunta"),
                "exclamacion": st.column_config.NumberColumn("¡!", format="percent", help="Titulares con exclamación"),
                "en_vivo": st.column_config.NumberColumn("EN VIVO", format="percent",
                                                         help="Titulares con EN VIVO, VIDEO u HOY")})
            if len(rasgos) == 2:
                nombres_rasgo = {"largo": "titula más largo", "con_cita": "usa más citas", "pregunta": "usa más preguntas",
                                 "exclamacion": "usa más exclamaciones", "en_vivo": "usa más marcas de inmediatez"}
                lideres = {m: [t for r, t in nombres_rasgo.items() if rasgos[r].idxmax() == m and rasgos[r].nunique() > 1]
                           for m in rasgos.index}
                st.caption(" · ".join(f"**{m}** {', '.join(v)}" for m, v in lideres.items() if v) + ".")

# ================================================================ 6. Datos
with tab_datos:
    st.markdown(f"### {len(filtrado)} noticias seleccionadas")
    tabla = filtrado[["titular", "medio_nombre", "categoria", "categoria_llm", "confianza", "fecha", "bajada", "url"]]
    st.dataframe(tabla, width="stretch", hide_index=True, height=480, column_config={
        "titular": st.column_config.TextColumn("titular", width="large"),
        "medio_nombre": "medio", "categoria": "sección del diario", "categoria_llm": "etiqueta LLM",
        "confianza": st.column_config.ProgressColumn("confianza LLM", format="%.2f", min_value=0, max_value=1),
        "fecha": st.column_config.DateColumn("fecha"),
        "url": st.column_config.LinkColumn("enlace", display_text="abrir")})
    st.download_button("⬇️ Descargar selección en CSV",
                       filtrado[["titular", "bajada", "medio", "seccion_original", "url", "fecha", "etiqueta",
                                 "etiqueta_llm", "confianza"]].to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

st.divider()
st.caption("La Portada Clasificada · Tarea 1 – Diplomatura Python Intermedio · Datos: larepublica.pe y elcomercio.pe "
           "(uso académico) · Modelo: DistilBERT multilingüe ajustado con etiquetas de qwen2.5:7b.")
