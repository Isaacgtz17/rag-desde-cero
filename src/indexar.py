"""Construye el índice: cargar -> chunking -> embeddings -> guardar en ChromaDB.

Se corre UNA vez (o cuando cambien los documentos o los parámetros de chunking).
Las preguntas después solo consultan la base; no vuelven a calcular nada.

Además guarda chunks.json + embeddings.npy (la versión de la etapa 2) para poder
comparar que Chroma devuelve lo mismo que nuestra búsqueda a mano con numpy.

Uso:  python src/indexar.py
"""
import json
import time

import chromadb
import numpy as np

import config
from cargar import cargar_documentos
from chunking import crear_chunks
from embeddings import embed_pasajes


def guardar_en_chroma(chunks: list[dict], vectores: np.ndarray):
    cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))   # base en disco
    # Empezar de cero en cada indexación: evita chunks viejos mezclados con nuevos
    if config.COLECCION in [c.name for c in cliente.list_collections()]:
        cliente.delete_collection(config.COLECCION)
    coleccion = cliente.create_collection(
        config.COLECCION,
        configuration={"hnsw": {"space": "cosine"}},   # distancia = 1 - similitud coseno
        embedding_function=None,                       # los vectores los calculamos nosotros (e5 + prefijos)
    )
    # Metadatos = todo menos el id y el texto (Chroma los guarda en campos propios)
    metadatos = [{k: v for k, v in c.items() if k not in ("id", "texto")} for c in chunks]
    coleccion.add(
        ids=[c["id"] for c in chunks],
        documents=[c["texto"] for c in chunks],
        embeddings=vectores.tolist(),
        metadatas=metadatos,
    )
    return coleccion.count()


if __name__ == "__main__":
    t0 = time.time()
    chunks = crear_chunks(cargar_documentos())
    print(f"{len(chunks)} chunks. Calculando embeddings...")
    vectores = embed_pasajes([c["texto"] for c in chunks])

    # Versión etapa 2 (para comparar)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(config.EMBEDDINGS_NPY, vectores)

    # Versión etapa 3
    n = guardar_en_chroma(chunks, vectores)
    print(f"Listo en {time.time() - t0:.0f}s. Chroma: {n} chunks en '{config.COLECCION}' ({config.CHROMA_DIR.name}/)")
