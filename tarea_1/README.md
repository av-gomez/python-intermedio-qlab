# 🗞️ Tarea 1 – Noticias peruanas: del titular al clasificador     

## 1. Pregunta del proyecto

> **¿Pueden convertir titulares peruanos en un clasificador de noticias?**

Se construyó un corpus comparado de titulares de **La República** y **El Comercio**, se etiquetó de forma semi-supervisada
con un LLM local, se ajustó un Transformer ligero (DistilBERT multilingüe) y se desplegó un dashboard en Streamlit que
clasifica titulares nuevos en vivo en 4 categorías: `politica`, `economia`, `deportes`, `ciencia_tecnologia`.       

**Respuesta corta:** **sí.** Con 400 titulares reales y etiquetas generadas por un LLM local, sin anotación manual masiva, un DistilBERT ajustado en pocos minutos de CPU alcanza **F1 macro = 0.92** en un test reservado, frente a **0.72** de la línea base zero-shot.

## 2. Medios y fechas de extracción

| | La República | El Comercio |
| :--- | :--- | :--- |
| **Opción** | C – Comparada | C – Comparada |
| **Noticias finales** | 200 (50 por categoría) | 200 (50 por categoría) |
| **Fecha de descarga** | 18/10/2026, 19:59 (UTC-5) | 18/10/2026, 19:47 (UTC-5) |
| **Rango de publicación** | 18/06/2023 – 18/10/2026 (78 % desde el 1/10/2026; 17 notas anteriores a sept. 2026 vienen de bloques "más noticias de la sección", sobre todo en ciencia) | 09/10/2026 – 18/10/2026 |
| **Fuentes** | RSS oficiales (`/rss/*.xml`), portadas de sección (JSON `__NEXT_DATA__`) y bloques "más noticias de la sección" de cada artículo | RSS oficiales (`/arc/outboundfeeds/rss/category/...`) y sitemaps web por sección + metadatos `og:` de cada artículo |

**Buenas prácticas de scraping:** se interpretó el `robots.txt` de cada medio (con comodines) y se validó cada URL antes de
pedirla; `User-Agent` descriptivo; pausa aleatoria de 1–2 s entre solicitudes; máximo 1 reintento; respaldo crudo en
`datos/raw/`. La República **prohíbe la paginación** (`Disallow: /*/pag/`), por lo que no se recorrieron páginas 2, 3, …

**Etiqueta de referencia:** la sección original del diario mapeada a las 4 categorías (tabla en `01_recoleccion.ipynb`).
Las noticias de otras secciones (sociedad, mundo, espectáculos, etc.) se descartaron y se reportaron.

## 3. Resultados

Todas las métricas finales se calcularon sobre el **conjunto de test reservado** (60 noticias, 15 % del total), que no se usó ni para entrenar ni para elegir el modelo.

| Modelo | Datos de evaluación | Accuracy | **F1 macro** | Latencia CPU |
| :--- | :--- | :---: | :---: | :---: |
| Zero-shot `Recognai/bert-base-spanish-wwm-cased-xnli` (línea base) | 40 titulares de test | 0.73 | **0.72** | cientos de ms |
| LLM débil `qwen2.5:7b` (Ollama) | test completo (60) | – | **0.95** | 5–6 s |
| **DistilBERT multilingüe fine-tuned con etiquetas del LLM** | test completo (60) | 0.92 | **0.92** | decenas de ms |

- **Auditoría humana del LLM:** 18/20 casos coinciden (**90 %**). Los 2 desacuerdos son noticias económicas con protagonistas políticos.
- **LLM frente a la sección del diario (400 noticias):** F1 macro de 0.93.
- **F1 por clase (DistilBERT, test):** deportes 0.93 · ciencia_tecnologia 0.93 · politica 0.91 · economia 0.90.
- **Modelo elegido para producción:** DistilBERT. Pierde ~3 puntos frente al LLM, pero es más de 50 veces más rápido, 9 veces más liviano y no necesita servidor externo. El LLM queda como anotador y como segunda opinión para los casos de baja confianza (cascada).

Para la entrega: `F1 Macro: 0.92 | Zero-shot: 0.72 | Acuerdo LLM-Humano: 18/20`

## 4. Hallazgos analíticos

1. **La frontera economía ↔ política concentra los errores, y el alumno hereda el sesgo del maestro.** El LLM etiquetó como
   `politica` 15 de las 100 noticias que el diario publicó en Economía (Petroperú, MEF, ministros, aranceles), y el DistilBERT
   entrenado con esas etiquetas reproduce el mismo patrón: economía es la clase con peor F1 (0.90) y política la de menor
   precisión (0.83). En la prensa peruana la economía se narra a través de actores estatales, y un titular con
   *personaje + declaración entre comillas* "suena" a política aunque trate de impuestos o crudo.
2. **Cada medio titula distinto según la sección.** La República titula más largo en general (mediana de 106 frente a 101
   caracteres) y sobre todo en ciencia (120 frente a 96), con titulares narrativos tipo curiosidad ("En 1986, la explosión del
   reactor de Chernóbil obligó…"). El Comercio titula más largo en política (112 frente a 96) y usa más preguntas (13 % de
   sus titulares frente a 4 %). Aun así, el clasificador rinde parecido en ambos (F1 0.93 en LR y 0.91 en EC con n≈30): el
   vocabulario temático pesa más que el estilo del medio.

## 5. Limitaciones técnicas

1. **La etiqueta de referencia es la sección del diario, no una anotación humana.** Las secciones son una aproximación
   ruidosa: notas como la reorganización del Minem (publicada en Economía) o la salud del papa (publicada en Política) son
   ambiguas. Además, el test tiene solo 60 noticias, así que cada error mueve el F1 ~1.7 puntos; las diferencias entre
   medios o entre LLM y DistilBERT (±3 puntos) no son estadísticamente concluyentes.
2. **Cobertura temporal y de fuentes limitada, y el modelo no sabe decir "ninguna".** El Comercio aporta solo 10 días de
   octubre de 2026 y La República está sesgada a las notas recientes, porque su `robots.txt` prohíbe la paginación. Por eso
   el modelo aprendió la coyuntura del momento (Keiko Fujimori, Liga 1 Clausura, Petroperú) y puede degradarse con otros
   temas. Además, siempre asigna una de las 4 categorías: un titular de espectáculos o policiales recibirá una etiqueta
   equivocada (la app avisa cuando la probabilidad es < 50 %). Por último, la confianza del LLM es autodeclarada y no está
   calibrada, y los pesos del modelo (541 MB) no caben en GitHub y deben regenerarse.

## 6. Cómo ejecutar en local

Requisitos: Python 3.12 (Windows, macOS o Linux). No se necesita GPU.

```bash
# 1. Entrar a la carpeta y crear un entorno virtual
cd tarea_1
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Instalar dependencias exactas (torch en versión CPU)
pip install -r requirements.txt

# 3. (Solo si no existe modelo/model.safetensors) regenerar el modelo
#    El notebook reutiliza datos/etiquetas_llm.csv, así que NO necesita Ollama para reentrenar.
jupyter nbconvert --to notebook --execute --inplace 02_clasificacion.ipynb

# 4. Lanzar el dashboard
streamlit run app.py
```

La app abre en `http://localhost:8501`.

**Pesos del modelo:** `modelo/model.safetensors` (~540 MB) supera el límite de 100 MB de GitHub y está en `.gitignore`.
En el repositorio quedan `config.json`, el tokenizer y `metricas.json`; el paso 3 regenera los pesos en ~5 min en CPU.
Para desplegar en Streamlit Community Cloud se puede subir el modelo a Hugging Face Hub y definir la variable de entorno
`MODELO_CLASIFICADOR=<usuario>/<repo>`.

**Para volver a etiquetar con el LLM** (opcional): instalar [Ollama](https://ollama.com), ejecutar `ollama pull qwen2.5:7b`,
borrar `datos/etiquetas_llm.csv` y ejecutar `02_clasificacion.ipynb`.

**Para volver a descargar las noticias** (opcional): borrar `datos/raw/*_crudo.json` y ejecutar `01_recoleccion.ipynb`
(~15 min por las pausas de cortesía). Con el respaldo presente, el notebook lo reutiliza.

## 7. Estructura

```text
tarea_1/
├── 01_recoleccion.ipynb      # Scraping, limpieza, mapeo y EDA (Plotly)
├── 02_clasificacion.ipynb    # Zero-shot, LLM débil, auditoría, split, fine-tuning, evaluación y comparativa
├── app.py                    # Dashboard "La Portada Clasificada": portada en vivo, juego, probador, resultados y corpus
├── .streamlit/config.toml    # Tema visual de la app
├── requirements.txt          # Dependencias exactas
├── README.md
├── .gitignore
├── datos/
│   ├── noticias.csv          # 400 noticias, 7 columnas
│   ├── etiquetas_llm.csv     # Etiquetas de qwen2.5:7b (titular, etiqueta_llm, confianza, modelo_llm, fecha, url)
│   ├── auditoria_humana.csv  # 20 casos auditados manualmente
│   ├── split.csv             # Partición train/val/test de cada noticia
│   └── raw/                  # Respaldo crudo (JSON, RSS, sitemaps, robots.txt)
└── modelo/                   # config.json, tokenizer, metricas.json (pesos excluidos de git)
```
