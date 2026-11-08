"""Paso 4 del RAG: buscar los chunks más relevantes para la pregunta.

Tres modos, de menor a mayor calidad (y costo):

1. "vectorial"       Embeddings + similitud coseno en Chroma (etapa 3).
                     Entiende significado ("levantar solicitud" ~ "abrir ticket"),
                     pero puede fallar con palabras exactas raras (nombres de sistemas, siglas).

2. "hibrida"         Vectorial + BM25 (búsqueda por palabras clave), combinadas con RRF.
                     BM25 premia chunks que contienen las palabras EXACTAS de la pregunta,
                     sobre todo las poco comunes en el corpus. Se complementa con la vectorial.

3. "hibrida+rerank"  Toma ~20 candidatos de la híbrida y un cross-encoder los reordena.
                     El cross-encoder lee pregunta y chunk JUNTOS (no por separado como los
                     embeddings), así que juzga mucho mejor la relevancia. Es lento, por eso
                     solo se aplica a unos pocos candidatos.

Uso para diagnosticar:
    python src/buscar.py "¿cómo me conecto a la VPN?"
    python src/buscar.py "¿cómo abro un ticket?" --manual manual_tickets.pdf
    python src/buscar.py "¿qué es cada etapa?" --modo vectorial
"""
import argparse
import re
import unicodedata
from functools import lru_cache

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config
from chunking import describir
from embeddings import embed_pregunta

MODOS = ["vectorial", "hibrida", "hibrida+rerank"]


# ---------------------------------------------------------------- Chroma
@lru_cache(maxsize=1)   # abrir la base una sola vez por proceso
def coleccion():
    cliente = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return cliente.get_collection(config.COLECCION)


def buscar_vectorial(pregunta: str, k: int, fuente: str | None = None) -> list[dict]:
    """Top-k por similitud coseno en Chroma. `score` = 1 - distancia."""
    r = coleccion().query(
        query_embeddings=[embed_pregunta(pregunta).tolist()],
        n_results=k,
        where={"fuente": fuente} if fuente else None,   # filtro por metadato
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": id_, "texto": doc, **meta, "score": 1 - dist}
        for id_, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


# ---------------------------------------------------------------- BM25
# Palabras tan comunes que no ayudan a distinguir chunks
STOPWORDS = set("""de la que el en y a los del se las por un para con no una su al lo como mas o pero sus le ya
este esta si porque entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante todos uno
les ni contra otros ese eso ante ellos e esto mi antes algunos que unos yo otro otras otra tanto esa estos mucho
cual poco ella estar estas algunas algo nosotros es son fue ser hace puede cada como cuantos cuantas debo puedo""".split())


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin acentos, sin stopwords. 'Cómo abro un Ticket' -> ['abro', 'ticket']."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")   # quita acentos
    return [t for t in re.findall(r"[a-z0-9]+", texto) if len(t) > 1 and t not in STOPWORDS]


@lru_cache(maxsize=1)
def indice_bm25():
    """BM25 se arma en memoria con los textos de Chroma (con ~500 chunks tarda milisegundos)."""
    datos = coleccion().get(include=["documents", "metadatas"])
    bm25 = BM25Okapi([tokenizar(d) for d in datos["documents"]])
    return bm25, datos


def buscar_bm25(pregunta: str, k: int, fuente: str | None = None) -> list[dict]:
    bm25, datos = indice_bm25()
    scores = bm25.get_scores(tokenizar(pregunta))
    if fuente:   # mismo filtro por manual que la vectorial
        scores = np.where([m["fuente"] == fuente for m in datos["metadatas"]], scores, -1)
    mejores = np.argsort(-scores)[:k]
    return [
        {"id": datos["ids"][i], "texto": datos["documents"][i], **datos["metadatas"][i], "score": float(scores[i])}
        for i in mejores if scores[i] > 0
    ]


# ---------------------------------------------------------------- Fusión RRF
def fusion_rrf(*listas: list[dict], k_rrf: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = suma de 1/(60 + lugar) en cada lista.

    Usa solo el LUGAR, no el score: así no hay que normalizar escalas distintas
    (coseno 0-1 vs BM25 0-30). Un chunk bien ubicado en ambas listas sube mucho.
    """
    acumulado, chunk_por_id = {}, {}
    for lista in listas:
        for lugar, c in enumerate(lista, start=1):
            acumulado[c["id"]] = acumulado.get(c["id"], 0) + 1 / (k_rrf + lugar)
            chunk_por_id[c["id"]] = c
    orden = sorted(acumulado, key=acumulado.get, reverse=True)
    return [{**chunk_por_id[i], "score": acumulado[i]} for i in orden]


def buscar_hibrida(pregunta: str, k: int, fuente: str | None = None) -> list[dict]:
    n = config.CANDIDATOS
    return fusion_rrf(buscar_vectorial(pregunta, n, fuente), buscar_bm25(pregunta, n, fuente))[:k]


# ---------------------------------------------------------------- Reranking
@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(config.MODELO_RERANKER, max_length=512)


def reranquear(pregunta: str, candidatos: list[dict], k: int) -> list[dict]:
    """El cross-encoder da un score de relevancia a cada par (pregunta, chunk)."""
    scores = reranker().predict([(pregunta, c["texto"]) for c in candidatos])
    orden = np.argsort(-scores)[:k]
    return [{**candidatos[i], "score": float(scores[i])} for i in orden]


# ---------------------------------------------------------------- Punto de entrada
def buscar(pregunta: str, k: int = config.TOP_K, fuente: str | None = None,
           modo: str = config.MODO_BUSQUEDA, reescribir: bool = config.REESCRIBIR_PREGUNTA) -> list[dict]:
    if modo == "vectorial":
        return buscar_vectorial(pregunta, k, fuente)
    if modo == "hibrida":
        return buscar_hibrida(pregunta, k, fuente)
    if modo == "hibrida+rerank":
        consultas = [pregunta]
        if reescribir:   # multi-query: original + variantes con otro vocabulario
            from reescribir import reescribir as variantes
            consultas += list(variantes(pregunta))
        # Una búsqueda híbrida por consulta, fusionadas con RRF -> candidatos
        candidatos = fusion_rrf(*[buscar_hibrida(c, config.CANDIDATOS, fuente) for c in consultas])
        candidatos = candidatos[: config.CANDIDATOS]
        # El reranker juzga contra la pregunta ORIGINAL (lo que el usuario realmente quiere)
        reordenados = reranquear(pregunta, candidatos, len(candidatos))
        if config.FUSIONAR_RERANK:
            # No confiar ciegamente en el reranker (modelo pequeño): combinar SU orden con el de
            # los candidatos. Medido en la evaluación: el reranker hundía chunks correctos que la
            # reescritura había subido al lugar 1-2 (pregunta coloquial vs. texto formal del manual).
            reordenados = fusion_rrf(reordenados, candidatos)
        return reordenados[:k]
    raise ValueError(f"modo desconocido: {modo}")


def manuales() -> list[str]:
    """Lista de manuales indexados (para filtros en la interfaz)."""
    return sorted({m["fuente"] for m in indice_bm25()[1]["metadatas"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pregunta", nargs="?", default="¿Cómo abro un ticket de soporte?")
    p.add_argument("--manual", help="buscar solo en este archivo")
    p.add_argument("--modo", choices=MODOS, default=config.MODO_BUSQUEDA)
    a = p.parse_args()
    print(f"Pregunta: {a.pregunta}   (modo: {a.modo})\n")
    for r in buscar(a.pregunta, fuente=a.manual, modo=a.modo):
        vista = r["texto"][:250].replace("\n", " ")
        print(f"[{r['score']:.3f}] {describir(r)}  ({r['id']})\n        {vista}...\n")
