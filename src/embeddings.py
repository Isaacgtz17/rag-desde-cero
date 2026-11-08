"""Paso 3 del RAG: convertir texto en vectores (embeddings).

Un embedding es una lista de números (768 para e5-base) que representa el
SIGNIFICADO del texto. Textos con significado parecido -> vectores cercanos,
aunque usen palabras distintas ("abrir ticket" ~ "levantar una solicitud").

Detalle importante de los modelos e5: fueron entrenados con prefijos.
  - A los documentos (chunks) se les antepone  "passage: "
  - A la pregunta del usuario se le antepone   "query: "
Si se te olvidan, el modelo funciona, pero recupera peor.
"""
from functools import lru_cache

import numpy as np
import transformers
from sentence_transformers import SentenceTransformer

import config

transformers.logging.set_verbosity_error()     # ocultar avisos internos de la librería
transformers.logging.disable_progress_bar()    # ocultar la barra "Loading weights"


@lru_cache(maxsize=1)   # cargar el modelo una sola vez (tarda unos segundos)
def modelo() -> SentenceTransformer:
    return SentenceTransformer(config.MODELO_EMBEDDINGS)


def embed_pasajes(textos: list[str]) -> np.ndarray:
    """Vectores de los chunks. Devuelve matriz (n_chunks, 768)."""
    return modelo().encode(
        ["passage: " + t for t in textos],
        normalize_embeddings=True,   # largo 1 -> coseno = producto punto (ver buscar.py)
        batch_size=16,
        show_progress_bar=True,
    )


def embed_pregunta(pregunta: str) -> np.ndarray:
    """Vector de la pregunta. Devuelve arreglo (768,)."""
    return modelo().encode("query: " + pregunta, normalize_embeddings=True)
