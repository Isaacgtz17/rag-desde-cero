"""Configuración central: rutas y parámetros en un solo lugar.

Si quieres experimentar (otro tamaño de chunk, otro modelo), cambia aquí
y no tienes que buscar números sueltos por todo el código.
"""
from pathlib import Path

# --- Rutas ---
RAIZ = Path(__file__).resolve().parent.parent      # carpeta del proyecto
DATA_RAW = RAIZ / "data" / "raw"                    # documentos originales (no se tocan)
DATA_PROCESSED = RAIZ / "data" / "processed"        # todo lo regenerable
OCR_CACHE = DATA_PROCESSED / "ocr"                  # texto de los PDFs ya pasados por OCR
CHUNKS_JSON = DATA_PROCESSED / "chunks.json"        # chunks con sus metadatos
EMBEDDINGS_NPY = DATA_PROCESSED / "embeddings.npy"  # matriz de vectores (un renglón por chunk)
CHROMA_DIR = RAIZ / "chroma_db"                     # base vectorial persistente (etapa 3)
COLECCION = "manuales"                              # nombre de la colección dentro de Chroma

# --- OCR (Tesseract) ---
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR = RAIZ / "tessdata"   # aquí está spa.traineddata
OCR_DPI = 200                      # más dpi = mejor lectura pero más lento

# --- Chunking ---
CHUNKING = "secciones"   # "tamano" (etapa 2) | "secciones" (etapa 4)
CHUNK_SIZE = 800      # [tamano] caracteres por chunk (~150-200 palabras)
CHUNK_OVERLAP = 150   # [tamano] caracteres repetidos entre chunks vecinos
CHUNK_MAX = 1300      # [secciones] máximo por chunk; secciones más largas se dividen por párrafos.
                      # e5 lee máx. 512 tokens: con 1500 había 3 chunks truncados (tablas: muchos tokens por "|")

# --- Embeddings ---
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"

# --- Búsqueda y LLM ---
TOP_K = 5                      # cuántos chunks se mandan al LLM
MODO_BUSQUEDA = "hibrida+rerank"   # "vectorial" | "hibrida" | "hibrida+rerank"  (etapa 4)
CANDIDATOS = 20                # cuántos trae cada buscador antes de fusionar/reranquear
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # cross-encoder multilingüe (~470 MB)
# Experimento post-etapa 5 (APAGADO): medido, no superó la línea base (ver NOTAS.md).
# La reescritura sí sube el chunk correcto en candidatos, pero el reranker lo vuelve a hundir.
# Vale la pena reactivarlo si se cambia a un reranker más fuerte.
REESCRIBIR_PREGUNTA = False    # multi-query: buscar también con variantes de la pregunta
MODELO_REESCRITURA = "claude-haiku-5-5"   # barato y rápido; solo reformula, no responde
FUSIONAR_RERANK = False        # combinar orden del reranker con el de los candidatos (RRF) en vez de reemplazarlo
MODELO_LLM = "claude-opus-5-5"

# --- Específico de TUS documentos (valores genéricos; personalízalos en config_local.py) ---
NOMBRE_APP = "Pregúntale a los manuales"

# Patrones (regex) de líneas de ruido propias de tus documentos, además de los genéricos de cargar.py.
# Ej.: la barra de menú de un sitio web que aparece en cada página de un PDF impreso desde el navegador.
PATRONES_RUIDO_EXTRA: list[str] = []

# Descripción breve de tus manuales, para la reescritura de preguntas (reescribir.py).
DESCRIPCION_CORPUS = "- Manuales de usuario de los sistemas internos de la organización."

# Preguntas de ejemplo que la interfaz muestra como botones al abrir.
PREGUNTAS_EJEMPLO = [
    "¿Cómo inicio sesión en el sistema?",
    "¿Qué hago si olvidé mi contraseña?",
    "¿Qué permisos tiene cada rol?",
    "¿Qué significa este mensaje de error?",
]

# Configuración privada (NO se sube a git): si existe src/config_local.py, sus valores reemplazan a los de arriba.
# Ahí van las rutas, descripciones y ejemplos propios de tus documentos.
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass
