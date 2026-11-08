# NOTAS — RAG desde cero

Bitácora de aprendizaje: qué hice en cada etapa, qué aprendí y qué cambiaría.
El corpus fueron 8 manuales de usuario de software interno en español (privados, no incluidos en el repo),
así que los ejemplos aquí están descritos de forma genérica.

---

## Etapa 1: Preparar el proyecto

### Qué hice
- Revisé el corpus antes de programar: 4 `.md` (~236 KB, encabezados `#/##/###`, muchas tablas) y
  4 `.pdf` (46 páginas) que resultaron ser **imágenes**: páginas web impresas a PDF, 0 caracteres extraíbles.
- Entorno virtual con Python 3.11 (evité 3.14 porque torch/chromadb pueden no tener wheels aún).
- Estructura: `data/raw` (originales, intocables), `data/processed` (regenerable), `src`, `eval`.
- `requirements.txt` que crece por etapa, con un comentario de para qué sirve cada librería.
- OCR con Tesseract + modelo de español (`spa.traineddata` de *tessdata_best*) en una carpeta `tessdata/` del
  proyecto (no requiere permisos de administrador).

### Qué aprendí
- **Revisar los datos antes de programar.** Un PDF puede "verse" con texto y no tenerlo: sin esa revisión,
  el RAG habría indexado 4 manuales vacíos sin dar ningún error.
- Separar datos de código: los originales no se modifican; todo lo derivado se puede borrar y regenerar.
- Anticipé el ruido: encabezados/pies del navegador en el OCR, secciones de índice y listas de figuras en los
  Markdown, y tablas que un chunking por tamaño partiría.

---

## Etapa 2: RAG mínimo a mano

### Qué hice
| Archivo | Paso | Detalle |
|---|---|---|
| `config.py` | — | Rutas y parámetros en un solo lugar |
| `cargar.py` | Carga | `.md` directo; `.pdf` → imagen 200 dpi → Tesseract. Caché del OCR **crudo**; la limpieza se aplica al cargar |
| `chunking.py` | Chunking | 800 caracteres, traslape 150, corte en salto de línea/espacio → 449 chunks |
| `embeddings.py` | Embeddings | `intfloat/multilingual-e5-base` (768 dim), prefijos `passage:` / `query:`, vectores normalizados |
| `indexar.py` | Índice | `chunks.json` + `embeddings.npy` |
| `buscar.py` | Búsqueda | Coseno = `vectores @ pregunta` (normalizados). Top-5. Herramienta de diagnóstico |
| `rag.py` | Generación | Prompt con contexto numerado → Claude → respuesta + fuentes |

### Qué aprendí
- **Caché de lo lento, limpieza de lo barato.** Al principio guardaba el OCR ya limpio: cada ajuste de limpieza
  obligaba a repetir ~105 s de OCR. Lo corregí guardando el OCR crudo.
- **e5 amontona los scores (0.80–0.90)**: el valor absoluto casi no dice nada; importa el orden y la distancia.
- **Primer fallo diagnosticado** (siempre revisar primero qué chunks se recuperaron): una pregunta con
  vocabulario coloquial dejó el chunk correcto en el lugar 7 (fuera del top-5); con el vocabulario del manual subió
  al lugar 2. Además, el índice del manual competía porque repite palabras de todo el documento.
- **El fallo era de recuperación, no del LLM**: Claude respondió "no encontré la descripción…" porque el chunk
  correcto nunca le llegó. Cambiar de LLM no lo arreglaría.
- Prompt RAG: responder solo con el contexto, admitir cuando no está, citar `[n]`. Con preguntas fuera de los
  manuales respondió "No encontré esa información" sin inventar.

---

## Etapa 3: Persistencia con ChromaDB y metadatos

### Qué hice
- ChromaDB 1.5 en disco, con `configuration={"hnsw": {"space": "cosine"}}` (distancia = 1 − similitud)
  y `embedding_function=None` (los vectores los calculo yo, con los prefijos de e5).
- Metadatos por chunk: `fuente`, `tipo` (md/pdf), `pagina` (PDF) y `seccion` (MD).
- Filtro por manual con `where={"fuente": ...}`; `buscar.py --comparar` contra la versión numpy.
- Interfaz preliminar en Streamlit (adelantada de la etapa 6).

### Qué aprendí
- **Validar la migración**: Chroma y numpy dieron el mismo top-5 con los mismos scores. Con ~450 chunks,
  el índice aproximado HNSW encuentra exactamente lo mismo que la fuerza bruta.
- El filtro por metadatos quita la ambigüedad entre manuales, pero **no arregla el orden dentro de un manual**.
- El chunking por tamaño partía tablas a la mitad y su metadato `seccion` podía mentir (un chunk que cruza
  dos secciones se etiqueta con la primera).

---

## Etapa 4: Mejorar la recuperación

### Qué hice
1. **Script de diagnóstico** (`eval/diagnostico.py`, sin LLM, $0): cada caso define qué TEXTO debe contener el
   chunk correcto (no su id, que cambia con el chunking). Medí la **línea base antes** de cambiar nada.
2. **Chunking por secciones**: un chunk por sección `#`–`####` (MD) o por página (PDF); secciones largas se dividen
   por **párrafos**, así una tabla nunca se corta. Cada chunk empieza con su ruta (`Manual X › 2. Capítulo › 2.2 Sección`).
   Descarté índice, lista de figuras, historial de revisiones, comentarios HTML y diagramas mermaid.
   449 → 297 chunks. `CHUNK_MAX = 1300` para no pasar de los 512 tokens de e5.
3. **Búsqueda híbrida**: BM25 (`rank-bm25`, sin acentos ni stopwords) + vectorial, fusionadas con **RRF**
   (`1/(60+lugar)`: usa lugares, no scores, así no hay que normalizar escalas distintas).
4. **Reranking**: cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` reordena 20 candidatos → top-5.

### Resultados (aciertos en top-5 de 6 casos difíciles)
| Configuración | vectorial | híbrida | híbrida+rerank |
|---|---|---|---|
| Chunking por tamaño | 3/6 | 3/6 | 4/6 |
| Secciones (con diagramas mermaid) | 3/6 | 3/6 | 5/6 |
| **Secciones sin mermaid** | 4/6 | 4/6 | **6/6** |

Tiempo por pregunta: híbrida ~0.1 s; con rerank ~4 s (CPU).

### Qué aprendí
- **Medir antes de cambiar.** Sin la línea base no sabría qué arregló cada pieza.
- **Cada mejora arregla un tipo de fallo distinto**: una tabla partida solo la arregla el chunking (ningún
  buscador encuentra un chunk que no existe); un chunk enterrado en el lugar 15 lo rescató el reranker (→ 3).
- **Mis propios cambios metieron ruido**: el chunking por secciones incluyó un diagrama en código dentro del chunk
  de una tabla y ese caso empeoró (lugar 7 → 22). Quitarlo: 22 → 8, y con rerank → 3.
- **La híbrida no siempre ayuda**: si la palabra coloquial del usuario aparece en otra sección del manual,
  BM25 empuja hacia la sección equivocada. El reranker lo corrige.
- Los scores del reranker no están en 0–1 (pueden ser negativos): solo importa el orden.

---

## Etapa 5: Evaluación

### Qué hice
- 26 preguntas **nuevas** (no usadas para ajustar nada), revisadas a mano: pregunta natural, respuesta esperada,
  fuente, `debe_contener`, tipo. 13 con vocabulario distinto al del manual; 2 fuera de alcance.
  Formato en `eval/preguntas.ejemplo.json`.
- `eval/evaluar.py`:
  - `--validar` ($0): cada `debe_contener` existe completo en un chunk del índice.
  - `--solo-recuperacion` ($0): ¿el chunk correcto llegó al top-5? + MRR.
  - `--control`: prueba del juez. Respuestas esperadas → 100%; respuestas vacías → 0%. Si no, el juez no sirve.
  - Completa: Claude Opus 5.5 responde; **juez Claude Sonnet 5.5** (otro modelo) compara contra la esperada con
    salida estructurada (`explicacion` + `veredicto`: correcta/parcial/incorrecta).
  - Cada fallo se clasifica en **recuperación** (el chunk no llegó) o **generación** (llegó y respondió mal).
  - Errores de API se marcan aparte: no cuentan como "incorrecta".
- Piloto de 3 preguntas para medir el costo real antes de correr las 26.

### Resultados
| Métrica | Valor |
|---|---|
| Recuperación: chunk correcto en top-5 | 20/24 = 83% (± 15%) |
| MRR | 0.71 |
| **Respuestas correctas** | **23/26 = 88% (± 12%)** |
| Fuera de alcance (no inventar) | 2/2 |
| Fallos de generación | **0** |
| Costo de la corrida completa | ~$0.54 (~8 min) |

### Qué aprendí
- **Sobreajuste medido**: en los 6 casos con los que ajusté, 100%; en preguntas nuevas, 83%.
- **Con pocas preguntas el número es ruidoso** (±12–15 puntos): una mejora de 3 puntos no se distingue del azar.
- **El cuello de botella es la recuperación, no el LLM**: 0 fallos de generación. Cuando el contexto llega,
  Claude responde bien; cuando no llega, dice "no encontré" en vez de inventar.
- **La diferencia de vocabulario es el patrón #1 de fallo** (término coloquial vs. término del manual).
- **El dataset también puede fallar**: un dato repetido en dos secciones contó como "recuperación ✗" aunque la
  respuesta fue correcta. `debe_contener` asume un único lugar correcto.
- Probar al juez antes de confiar en él y leer sus explicaciones.

### Experimento: reescritura de la pregunta (multi-query) — no entró
- `src/reescribir.py`: Haiku 5.5 genera 3 variantes de la pregunta con vocabulario formal; busco con todas,
  fusiono con RRF y el reranker ordena contra la pregunta original. El prompt solo describe el corpus (sin
  términos sacados de los fallos, para no hacer trampa con el examen).

| Versión | Eval top-5 | MRR | Diagnóstico | s/pregunta |
|---|---|---|---|---|
| **Base** | **20/24** | **0.71** | **6/6** | **~3.6** |
| + reescritura | 20/24 | 0.71 | 6/6 | ~6.3 |
| + reescritura + fusionar rerank | 20/24 | 0.67 | 5/6 | ~6.3 |

- **Diagnóstico por etapa** (lo más valioso): la reescritura SÍ subía el chunk correcto en los candidatos
  (lugar 12 → 2, 15 → 2), pero el reranker los volvía a hundir (→ 16, 11) porque compara contra la pregunta coloquial.
- **Decisión**: un cambio que no supera la línea base no entra. Queda apagado en `config.py`. El verdadero
  cuello de botella es el reranker.

---

## Etapa 6: Interfaz con Streamlit

### Qué hice
- Preguntas de ejemplo como botones, chat con historial, botón de nueva conversación.
- Fuentes legibles (`archivo · § sección` / `archivo · pág. N`).
- "Mostrar detalles técnicos": id, score y texto exacto de cada chunk (diagnóstico sin salir de la interfaz).
- 👍/👎 por respuesta → `eval/feedback.jsonl`: las 👎 son candidatas a nuevas preguntas de evaluación (reales).
- Filtro por manual, tiempo de respuesta, errores de API con mensaje claro; modelos precargados con `st.cache_resource`.

### Qué aprendí
- Streamlit re-ejecuta el script completo en cada interacción: estado en `st.session_state`, objetos pesados en
  `st.cache_resource`.
- **Bug encontrado al probar en el navegador**: la "magia" de Streamlit muestra cualquier expresión suelta, y la
  línea que precargaba los modelos los imprimió en pantalla. `cache_resource` además repite lo que la función mostró
  la primera vez, así que corregir el código no bastó: hubo que reiniciar. Solución de raíz: `magicEnabled = false`.

---

## Resumen

```
PDF (imagen) ──OCR──┐
                    ├─> limpieza ─> chunks por sección ─> embeddings e5 ─> ChromaDB
Markdown ───────────┘                                         │
                                                              ▼
pregunta ─> [vectorial + BM25] ─RRF─> 20 candidatos ─> reranker ─> top-5 ─> Claude ─> respuesta + fuentes
```

**Las 5 lecciones más importantes:**
1. **Revisar los datos antes de programar** (los PDF no tenían texto).
2. **Cuando algo falla, primero mirar qué chunks se recuperaron**: casi siempre el problema es la recuperación.
3. **Medir antes y después de cada cambio**, con preguntas que no usaste para ajustar.
4. **Diagnosticar por etapa**, no solo el número final.
5. **Un cambio que no supera la línea base no entra**, aunque suene a mejora.

**Siguientes pasos**: un reranker más fuerte (p. ej. `BAAI/bge-reranker-v2-m3`) y volver a probar la reescritura;
respuestas en streaming; convertir los 👎 en preguntas de evaluación.
