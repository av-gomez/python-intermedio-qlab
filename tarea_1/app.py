"""Dashboard Streamlit – Tarea 1: Noticias peruanas, del titular al clasificador.

Ejecutar desde la carpeta tarea_1/:
    streamlit run app.py
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # descarga HTTP clásica si el modelo viene de Hugging Face Hub

BASE = Path(__file__).parent
# Carpeta local con el modelo entrenado en 02_clasificacion.ipynb, o un repo de Hugging Face Hub
MODELO = os.environ.get("MODELO_CLASIFICADOR", str(BASE / "modelo"))

CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {"politica": "Política", "economia": "Economía", "deportes": "Deportes",
           "ciencia_tecnologia": "Ciencia y tecnología"}
ORDEN = [NOMBRES[c] for c in CATEGORIAS]
# Paleta categórica de referencia (posiciones 1-4, orden fijo): el color sigue a la categoría en toda la app
COLOR_CAT = {"Política": "#2a78d6", "Economía": "#eb6834", "Deportes": "#1baf7a", "Ciencia y tecnología": "#eda100"}
# Los medios usan el violeta (posición 7, libre) y un gris medio: no compiten con las categorías
# y se leen tanto en tema claro como oscuro
MEDIOS = {"La Republica": "La República", "El Comercio": "El Comercio"}
COLOR_MEDIO = {"La República": "#7b6ce0", "El Comercio": "#8a8984"}
GRIS, DESTACADO = "#9c9b95", "#2a78d6"

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

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🗞️", layout="wide")


# ================================================================ Datos y modelo
@st.cache_data
def cargar_datos():
    noticias = pd.read_csv(BASE / "datos" / "noticias.csv")
    llm = pd.read_csv(BASE / "datos" / "etiquetas_llm.csv")[["url", "etiqueta_llm", "confianza"]]
    df = noticias.merge(llm, on="url", how="left")
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


@st.cache_resource(show_spinner="Cargando el modelo DistilBERT…")
def cargar_clasificador():
    from transformers import pipeline
    return pipeline("text-classification", model=MODELO, tokenizer=MODELO, top_k=None, device=-1)


@st.cache_resource
def indice_similitud(titulares):
    vectorizador = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True,
                                   stop_words=sorted(STOPWORDS))
    return vectorizador, vectorizador.fit_transform(titulares)


def modelo_disponible():
    local = Path(MODELO)
    return not local.exists() or any(local.glob("*.safetensors")) or any(local.glob("*.bin"))


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
                      hoverlabel=dict(font_size=13), **({"height": alto} if alto else {}))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


df = cargar_datos()
metricas = cargar_metricas()

# ================================================================ Encabezado
st.title("¿Pueden convertir titulares peruanos en un clasificador de noticias?")
if metricas:
    st.markdown(
        f"**Sí.** Con {len(df)} titulares de *La República* y *El Comercio*, etiquetados por un LLM local "
        f"y sin anotación manual masiva, un DistilBERT ajustado clasifica titulares nuevos en 4 categorías con "
        f"**F1 macro = {metricas['test_f1_macro']:.2f}** en un test reservado, frente a "
        f"{metricas['zero_shot_f1_macro']:.2f} de un modelo sin entrenar."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Noticias en el corpus", f"{len(df)}", "200 por medio · 50 por categoría", delta_color="off",
              help="Recolectadas el 18/10/2026 de RSS, portadas de sección y sitemaps, respetando robots.txt.")
    k2.metric("F1 macro en test", f"{metricas['test_f1_macro']:.2f}",
              f"{metricas['test_f1_macro'] - metricas['zero_shot_f1_macro']:+.2f} vs. sin entrenar",
              help="DistilBERT multilingüe ajustado, evaluado en 60 noticias que nunca vio al entrenar.")
    if metricas.get("auditoria_acuerdo_pct") is not None:
        k3.metric("Acuerdo LLM vs. revisión humana", f"{metricas['auditoria_acuerdo_pct']:.0f}%",
                  f"{metricas['auditoria_acuerdos']} de {metricas['auditoria_total']} casos", delta_color="off",
                  help="Una persona revisó 20 etiquetas del LLM elegidas al azar.")
    k4.metric("Velocidad del clasificador", f"~{metricas['latencia_ms']['fine_tuned']} ms",
              f"el LLM tarda ~{metricas['latencia_ms']['llm'] / 1000:.0f} s", delta_color="off",
              help="Tiempo por titular en CPU, medido en 02_clasificacion.ipynb.")

with st.expander("Cómo se construyó"):
    st.markdown("""
1. **Recolección** – 400 noticias (opción C, comparada): RSS oficiales, portadas de sección y sitemaps,
   con `robots.txt` respetado y pausas de 1–2 s. La **etiqueta de referencia** es la sección del diario
   mapeada a 4 categorías; las noticias de otras secciones se descartaron.
2. **Línea base** – modelo zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli` sin entrenamiento.
3. **Supervisión débil** – `qwen2.5:7b` (Ollama, local) etiqueta las 400 noticias; una persona audita 20 al azar.
4. **Fine-tuning** – `distilbert-base-multilingual-cased` entrenado con las etiquetas del LLM
   (split estratificado 70/15/15, `random_state=42`) y evaluado contra la sección del diario.
""")

# ================================================================ Filtros
st.sidebar.header("Filtros")
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

tab_probar, tab_modelo, tab_corpus, tab_datos = st.tabs(
    ["🧪 Probar el clasificador", "📈 Resultados del modelo", "🔍 Explorar el corpus", "📄 Datos"])

# ================================================================ 1. Probador en vivo
with tab_probar:
    st.subheader("Escribe un titular y el modelo predice su categoría")
    ejemplos = ["Congreso aprueba nuevo retiro de fondos de AFP",
                "Universitario vence a Alianza Lima en el clásico y se acerca al título",
                "NASA confirma hallazgo de agua en un asteroide cercano a la Tierra",
                "BCR mantiene su tasa de referencia por tercer mes consecutivo",
                "Ministro de Economía se reúne con congresistas para discutir el presupuesto"]
    elegido = st.pills("Ejemplos para probar", ejemplos, selection_mode="single")
    with st.form("probador"):
        titular = st.text_area("Titular", value=elegido or "", height=80,
                               placeholder="Pega aquí un titular de cualquier diario peruano…")
        enviado = st.form_submit_button("Clasificar", type="primary")

    if enviado:
        if not titular.strip():
            st.warning("Escribe un titular primero.")
        elif not modelo_disponible():
            st.error("No se encontraron los pesos del modelo en `modelo/`. Ejecuta `02_clasificacion.ipynb` "
                     "o define la variable de entorno `MODELO_CLASIFICADOR` con un repo de Hugging Face.")
        else:
            salida = cargar_clasificador()(titular.strip())
            salida = salida[0] if isinstance(salida[0], list) else salida
            probs = pd.DataFrame(salida).rename(columns={"label": "etiqueta", "score": "probabilidad"})
            probs["categoria"] = probs["etiqueta"].map(NOMBRES)
            probs = probs.set_index("categoria").loc[ORDEN].reset_index()
            mejor = probs.loc[probs["probabilidad"].idxmax()]

            r1, r2 = st.columns([2, 3], gap="large")
            with r1:
                st.metric("Categoría predicha", mejor["categoria"], f"{mejor['probabilidad']:.0%} de probabilidad",
                          delta_color="off")
                segunda = probs.drop(index=probs["probabilidad"].idxmax()).nlargest(1, "probabilidad").iloc[0]
                if mejor["probabilidad"] < 0.6:
                    st.warning(f"**Confianza baja.** Duda con *{segunda['categoria']}* "
                               f"({segunda['probabilidad']:.0%}). Puede ser un titular ambiguo o de un tema "
                               "fuera de las 4 categorías (el modelo siempre elige una).")
                else:
                    st.success(f"Confianza alta. Segunda opción: {segunda['categoria']} ({segunda['probabilidad']:.0%}).")
            with r2:
                fig = go.Figure(go.Bar(
                    x=probs["probabilidad"], y=probs["categoria"], orientation="h",
                    marker_color=[COLOR_CAT[c] if c == mejor["categoria"] else GRIS for c in probs["categoria"]],
                    text=probs["probabilidad"].map("{:.0%}".format), textposition="outside",
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                fig.update_layout(title="Probabilidad por categoría", xaxis=dict(range=[0, 1.12], tickformat=".0%"),
                                  yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]))
                st.plotly_chart(estilo(fig, 260), width="stretch")

            st.markdown("**Titulares más parecidos del corpus** (similitud de vocabulario, TF-IDF)")
            vectorizador, matriz = indice_similitud(tuple(df["titular"]))
            sim = cosine_similarity(vectorizador.transform([titular]), matriz).ravel()
            vecinos = df.assign(similitud=sim).nlargest(5, "similitud")
            vecinos = vecinos[vecinos["similitud"] > 0]
            if vecinos.empty:
                st.caption("Ningún titular del corpus comparte vocabulario con el texto ingresado.")
            else:
                st.dataframe(vecinos[["titular", "medio_nombre", "categoria", "similitud", "url"]],
                             hide_index=True, width="stretch",
                             column_config={"medio_nombre": "medio", "categoria": "sección",
                                            "similitud": st.column_config.ProgressColumn(
                                                "similitud", format="%.2f", min_value=0, max_value=1),
                                            "url": st.column_config.LinkColumn("enlace", display_text="abrir")})

# ================================================================ 2. Resultados del modelo
with tab_modelo:
    if not metricas:
        st.info("Ejecuta `02_clasificacion.ipynb` para generar `modelo/metricas.json`.")
    else:
        st.subheader("Tres formas de clasificar, comparadas en el test reservado")
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
                                   customdata=comp["nota"], hovertemplate="%{y}<br>F1 macro %{x:.3f}<br>%{customdata}<extra></extra>"))
            fig.update_layout(title="Calidad (F1 macro, más es mejor)", xaxis=dict(range=[0, 1.1]))
            st.plotly_chart(estilo(fig, 260), width="stretch")
        with c2:
            fig = go.Figure(go.Bar(x=comp["ms"], y=comp["modelo"], orientation="h", marker_color=colores,
                                   text=[f"{v:,.0f} ms" for v in comp["ms"]], textposition="outside",
                                   hovertemplate="%{y}<br>%{x:,.0f} ms por titular<extra></extra>"))
            fig.update_layout(title="Tiempo por titular en CPU (escala logarítmica, menos es mejor)",
                              xaxis=dict(type="log", range=[1, math.log10(lat["llm"]) + 0.6],
                                         tickvals=[10, 100, 1000, 10000], ticktext=["10 ms", "100 ms", "1 s", "10 s"]))
            st.plotly_chart(estilo(fig, 260), width="stretch")
        st.caption(f"El LLM es el más preciso, pero DistilBERT logra casi lo mismo siendo "
                   f"~{lat['llm'] / lat['fine_tuned']:.0f} veces más rápido y sin servidor externo: por eso es el "
                   "modelo del probador. El LLM queda como anotador para re-entrenar y para casos dudosos.")

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
            fig = go.Figure(go.Bar(x=f1.values, y=f1.index, orientation="h",
                                   marker_color=[COLOR_CAT[c] for c in f1.index],
                                   text=[f"{v:.2f}" for v in f1.values], textposition="outside",
                                   hovertemplate="%{y}: F1 %{x:.3f}<extra></extra>"))
            fig.update_layout(title="F1 por categoría", xaxis=dict(range=[0, 1.12]),
                              yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]))
            st.plotly_chart(estilo(fig, 260), width="stretch")
            st.markdown("**Lectura:** deportes y ciencia tienen un vocabulario propio (clubes, marcadores, IA, "
                        "científicos). La frontera difícil es **economía ↔ política**: la economía peruana se cuenta "
                        "a través de ministros, Petroperú o el Gobierno, y el modelo heredó esa confusión del LLM.")

# ================================================================ 3. Explorar el corpus
with tab_corpus:
    if filtrado.empty:
        st.info("No hay noticias con los filtros actuales.")
    else:
        st.subheader("¿El LLM etiqueta igual que el diario?")
        dist = (filtrado.groupby(["categoria", "categoria_llm"]).size().rename("noticias").reset_index())
        dist["%"] = dist["noticias"] / dist.groupby("categoria")["noticias"].transform("sum")
        acuerdo = filtrado.groupby("categoria")["coincide_llm"].mean()
        fig = px.bar(dist, x="%", y="categoria", color="categoria_llm", orientation="h",
                     color_discrete_map=COLOR_CAT, custom_data=["noticias", "categoria_llm"],
                     category_orders={"categoria": [c for c in ORDEN if c in acuerdo.index],
                                      "categoria_llm": ORDEN},
                     labels={"categoria": "Sección del diario", "categoria_llm": "Etiqueta del LLM", "%": ""})
        fig.update_traces(marker_line_color="rgba(255,255,255,0.7)", marker_line_width=1.5,
                          hovertemplate="Sección: %{y}<br>LLM dijo: %{customdata[1]}<br>%{customdata[0]} noticias (%{x:.0%})<extra></extra>")
        for cat, valor in acuerdo.items():
            fig.add_annotation(x=1.01, y=cat, text=f"{valor:.0%} coincide", showarrow=False, xanchor="left")
        fig.update_layout(title="Distribución de las etiquetas del LLM dentro de cada sección",
                          xaxis=dict(tickformat=".0%", range=[0, 1.18]), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(estilo(fig, 320), width="stretch")
        st.caption("Cada barra es una sección del diario; los colores muestran cómo la etiquetó el LLM. "
                   "Activa *Solo noticias donde el LLM no coincide* en la barra lateral para leer esos casos en **Datos**.")

        st.divider()
        st.subheader("Palabras que distinguen cada categoría")
        palabras = palabras_distintivas(filtrado)
        if palabras.empty:
            st.caption("Hay muy pocas noticias seleccionadas para calcular palabras distintivas.")
        else:
            cats_presentes = [c for c in ORDEN if c in palabras["categoria"].unique()]
            columnas = st.columns(len(cats_presentes))
            for col, cat in zip(columnas, cats_presentes):
                p = palabras[palabras["categoria"] == cat].sort_values("titulares")
                fig = go.Figure(go.Bar(x=p["titulares"], y=p["palabra"], orientation="h", marker_color=COLOR_CAT[cat],
                                       hovertemplate="«%{y}» aparece en %{x} titulares<extra></extra>"))
                fig.update_layout(title=cat, xaxis_title="titulares")
                col.plotly_chart(estilo(fig, 320), width="stretch")
            st.caption("Palabras más características de cada categoría frente a las demás (log-odds suavizado; "
                       "solo palabras presentes en 3 o más titulares). Filtra por medio para comparar el vocabulario "
                       "de La República y El Comercio.")

        st.divider()
        st.subheader("¿Titulan distinto La República y El Comercio?")
        c1, c2 = st.columns([3, 2], gap="large")
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
                                         marker=dict(size=13, color=color, line=dict(color="white", width=2)),
                                         hovertemplate=f"{medio}<br>%{{y}}: mediana %{{x:.0f}} caracteres<extra></extra>"))
            fig.update_layout(title="Longitud mediana del titular (caracteres)",
                              yaxis=dict(categoryorder="array", categoryarray=ORDEN[::-1]),
                              legend=dict(orientation="h", y=-0.2))
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
            st.markdown("**Rasgos de estilo** (sobre las noticias filtradas)")
            st.dataframe(rasgos, width="stretch", column_config={
                "largo": st.column_config.NumberColumn("largo mediano", format="%d car."),
                "con_cita": st.column_config.NumberColumn("con cita", format="percent"),
                "pregunta": st.column_config.NumberColumn("pregunta", format="percent"),
                "exclamacion": st.column_config.NumberColumn("exclamación", format="percent"),
                "en_vivo": st.column_config.NumberColumn("EN VIVO / VIDEO / HOY", format="percent")})
            if len(rasgos) == 2:
                nombres_rasgo = {"largo": "titula más largo", "con_cita": "usa más citas", "pregunta": "usa más preguntas",
                                 "exclamacion": "usa más exclamaciones", "en_vivo": "usa más marcas de inmediatez"}
                lideres = {m: [t for r, t in nombres_rasgo.items() if rasgos[r].idxmax() == m and rasgos[r].nunique() > 1]
                           for m in rasgos.index}
                st.caption(" · ".join(f"**{m}** {', '.join(v)}" for m, v in lideres.items() if v) + ".")

# ================================================================ 4. Datos
with tab_datos:
    st.subheader(f"{len(filtrado)} noticias seleccionadas")
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
st.caption("Tarea 1 – Diplomatura Python Intermedio · Datos: larepublica.pe y elcomercio.pe (uso académico) · "
           "Modelo: DistilBERT multilingüe ajustado con etiquetas de qwen2.5:7b.")
