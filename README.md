# RAG desde cero

Sistema de preguntas y respuestas sobre documentos propios (RAG) en español, construido **a mano** —sin
LangChain ni LlamaIndex— para entender cada pieza: OCR, chunking, embeddings, búsqueda híbrida, reranking,
generación con Claude, evaluación e interfaz web.

El proceso completo, con mediciones y lecciones de cada etapa, está en [NOTAS.md](NOTAS.md).

```
PDF (imagen) ──OCR──┐
                    ├─> limpieza ─> chunks por sección ─> embeddings e5 ─> ChromaDB
Markdown ───────────┘                                         │
                                                              ▼
pregunta ─> [vectorial + BM25] ─RRF─> 20 candidatos ─> reranker ─> top-5 ─> Claude ─> respuesta + fuentes
```

## Resultados (sobre 8 manuales de usuario privados)
| | |
|---|---|
| Respuestas correctas (26 preguntas nuevas, juez LLM) | **88%** (± 12%) |
| Chunk correcto en el top-5 | 83% |
| Preguntas fuera de los documentos | 2/2 respondidas con "no encontré" (sin inventar) |
| Fallos de generación | 0 (todos los fallos fueron de recuperación) |

## Características
- **Carga**: Markdown directo y PDF escaneados con OCR (Tesseract, español), con caché y limpieza de ruido.
- **Chunking por secciones**: respeta encabezados y nunca parte tablas; cada chunk lleva su ruta (`Manual › Capítulo › Sección`).
- **Embeddings** multilingües locales (`intfloat/multilingual-e5-base`) en **ChromaDB** con metadatos (fuente, sección, página).
- **Búsqueda híbrida** BM25 + vectorial con Reciprocal Rank Fusion, y **reranking** con cross-encoder.
- **Generación** con Claude: responde solo con el contexto, cita fuentes `[n]` y admite cuando no sabe.
- **Evaluación**: recuperación (gratis) y respuestas con un LLM como juez, diagnóstico de cada fallo y control del juez.
- **Interfaz** Streamlit: fuentes, filtro por documento, modo diagnóstico y 👍/👎.

## Requisitos (Windows)
- Python 3.11
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki): `winget install -e --id UB-Mannheim.TesseractOCR`
- Modelo de español en `tessdata/`: copia `eng.traineddata` y `osd.traineddata` desde
  `C:\Program Files\Tesseract-OCR\tessdata\` y descarga
  [`spa.traineddata`](https://github.com/tesseract-ocr/tessdata_best/raw/main/spa.traineddata).
- API key de Anthropic: copia `.env.example` como `.env` y pon tu `ANTHROPIC_API_KEY`.

## Instalación
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso
1. Pon tus documentos (`.md` o `.pdf`) en `data/raw/`.
2. (Opcional) Crea `src/config_local.py` para personalizar el nombre de la app, las preguntas de ejemplo,
   la descripción de tus documentos y patrones de ruido propios. Ver la sección final de `src/config.py`.
3. Indexa y abre la interfaz:
```powershell
python src\indexar.py          # OCR + chunks + embeddings + ChromaDB (repetir si cambian los documentos)
streamlit run src\app.py       # http://localhost:8501
python src\rag.py "tu pregunta"     # desde la terminal
python src\buscar.py "tu pregunta"  # ver qué chunks se recuperan (diagnóstico, sin costo)
```

## Evaluación
Copia las plantillas y escribe preguntas sobre **tus** documentos:
`eval/preguntas.ejemplo.json` → `eval/preguntas.json` y `eval/casos_diagnostico.ejemplo.json` → `eval/casos_diagnostico.json`.
```powershell
python eval\evaluar.py --validar             # revisa que las preguntas estén bien armadas
python eval\evaluar.py --solo-recuperacion   # ¿llega el chunk correcto? (sin costo)
python eval\evaluar.py --control             # prueba al juez (esperadas ~100%, vacías ~0%)
python eval\evaluar.py                       # evaluación completa con juez LLM
python eval\diagnostico.py                   # casos difíciles × 3 modos de búsqueda (sin costo)
```

## Estructura
```
src/
  config.py      parámetros (modelos, tamaños, modo de búsqueda) + config_local.py opcional
  cargar.py      lectura de .md y OCR de .pdf (con caché)
  chunking.py    chunks por sección (MD) / por página (PDF), con su ruta
  embeddings.py  intfloat/multilingual-e5-base
  indexar.py     construye la base ChromaDB
  buscar.py      vectorial, BM25, híbrida (RRF) y reranking
  reescribir.py  reescritura de pregunta (experimento, apagado)
  rag.py         prompt + Claude + fuentes
  app.py         interfaz Streamlit
eval/
  evaluar.py     evaluación de recuperación y respuestas
  diagnostico.py diagnóstico de recuperación por modo
  *.ejemplo.json plantillas de preguntas y casos
```
