"""Paso 2 del RAG: partir cada documento en chunks.

Dos estrategias (se elige en config.CHUNKING):

1. "tamano" (etapa 2): trozos de ~800 caracteres con traslape. Simple, pero ignora
   la estructura: parte tablas a la mitad y un chunk puede mezclar dos secciones.

2. "secciones" (etapa 4): respeta la estructura del documento.
   - Markdown: un chunk por sección (#, ##, ###, ####). Si una sección es muy larga,
     se divide por PÁRRAFOS (líneas en blanco), así una tabla nunca se corta.
   - PDF: un chunk por página (cada página del manual es una pantalla con su explicación).
   - Cada chunk lleva al inicio su "ruta" (manual › capítulo › sección). Eso le da
     contexto al embedding y al BM25: un chunk que solo dice "Pasos: 1. Abra la tarjeta"
     no sabe de qué habla; con la ruta, sí.
   - Se descartan secciones de ruido: índice, lista de figuras, historial de revisiones...

Uso para probar:  python src/chunking.py
"""
import re
import unicodedata

import config
from cargar import cargar_documentos

RE_ENCABEZADO = re.compile(r"^(#{1,4})\s+(.+?)\s*$")   # "## 2.2 Etapas del proceso" -> nivel 2
RE_PAGINA = re.compile(r"\[Página (\d+)\]")            # marca que pone cargar.py en los PDFs
RE_COMENTARIO = re.compile(r"<!--.*?-->", re.DOTALL)   # <!-- FIG:... --> marcas de figuras
# Diagramas como código (A["NUEVO"] --> B[...]): repiten lo que dice el texto y meten ruido al embedding
RE_MERMAID = re.compile(r"```mermaid.*?```", re.DOTALL)

# Secciones que no responden preguntas, pero repiten vocabulario de todo el manual
SECCIONES_RUIDO = ("indice", "lista de figuras", "historial de revisiones",
                   "control documental", "identificacion del documento")


def sin_acentos(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t.lower()) if unicodedata.category(c) != "Mn")


def nombre_manual(fuente: str) -> str:
    """'Manual_Usuario_Ventas.md' -> 'Manual Ventas';  'manual_soporte.pdf' -> 'Manual soporte'."""
    base = fuente.rsplit(".", 1)[0]
    base = re.sub(r"(?i)^(manual_usuario_|manual_)", "", base)
    return f"Manual {base}"


# ---------------------------------------------------------------- Estrategia 1: por tamaño
def partir_texto(texto: str, tamano: int, traslape: int) -> list[tuple[int, str]]:
    """Corta el texto en trozos de ~`tamano` caracteres, repitiendo `traslape` entre vecinos.

    Devuelve pares (posición_de_inicio, trozo). Corta en salto de línea o espacio
    para no partir palabras.
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):
            mitad = inicio + tamano // 2
            corte = texto.rfind("\n", mitad, fin)
            if corte == -1:
                corte = texto.rfind(" ", mitad, fin)
            if corte != -1:
                fin = corte
        trozo = texto[inicio:fin].strip()
        if trozo:
            chunks.append((inicio, trozo))
        if fin >= len(texto):
            break
        inicio = max(fin - traslape, inicio + 1)
        espacio = texto.find(" ", inicio, fin)
        if espacio != -1:
            inicio = espacio + 1
    return chunks


def ultima_coincidencia(patron: re.Pattern, texto: str, hasta: int):
    ultimo = None
    for m in patron.finditer(texto, 0, hasta):
        ultimo = m
    return ultimo


def chunks_por_tamano(doc: dict) -> list[dict]:
    texto, es_pdf = doc["texto"], doc["fuente"].lower().endswith(".pdf")
    salida = []
    for inicio, trozo in partir_texto(texto, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
        hasta = texto.find("\n", inicio)
        hasta = len(texto) if hasta == -1 else hasta
        if es_pdf:
            m = ultima_coincidencia(RE_PAGINA, texto, hasta)
            salida.append({"texto": trozo, "pagina": int(m.group(1)) if m else 1})
        else:
            m = ultima_coincidencia(re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE), texto, hasta)
            salida.append({"texto": trozo, "seccion": m.group(1).strip() if m else "(inicio)"})
    return salida


# ---------------------------------------------------------------- Estrategia 2: por secciones
def empacar_parrafos(cuerpo: str, maximo: int) -> list[str]:
    """Junta párrafos (separados por línea en blanco) en trozos de hasta `maximo` caracteres.

    Una tabla Markdown no tiene líneas en blanco dentro, así que es UN párrafo y queda entera.
    Solo si un párrafo solo excede el máximo se corta por tamaño (último recurso).
    """
    trozos, actual = [], ""
    for parrafo in re.split(r"\n\s*\n", cuerpo):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        if len(parrafo) > maximo:                       # párrafo gigante: cortar por tamaño
            if actual:
                trozos.append(actual)
                actual = ""
            trozos += [t for _, t in partir_texto(parrafo, maximo, config.CHUNK_OVERLAP)]
        elif len(actual) + len(parrafo) + 2 > maximo:   # no cabe: cerrar el trozo actual
            trozos.append(actual)
            actual = parrafo
        else:
            actual = f"{actual}\n\n{parrafo}" if actual else parrafo
    if actual:
        trozos.append(actual)
    return trozos


def secciones_markdown(texto: str) -> list[tuple[str, str]]:
    """Recorre el Markdown y devuelve (ruta_de_encabezados, cuerpo) por sección."""
    texto = RE_MERMAID.sub("", RE_COMENTARIO.sub("", texto))
    secciones, pila, cuerpo, en_codigo = [], [], [], False

    def cerrar():
        if pila and "\n".join(cuerpo).strip():
            secciones.append((" › ".join(t for _, t in pila), "\n".join(cuerpo).strip()))

    for linea in texto.splitlines():
        if linea.strip().startswith("```"):            # dentro de bloques de código "#" no es encabezado
            en_codigo = not en_codigo
        m = None if en_codigo else RE_ENCABEZADO.match(linea)
        if m:
            cerrar()
            nivel, titulo = len(m.group(1)), m.group(2).replace("*", "")
            while pila and pila[-1][0] >= nivel:       # salir de secciones del mismo nivel o más profundas
                pila.pop()
            pila.append((nivel, titulo))
            cuerpo = []
        else:
            cuerpo.append(linea)
    cerrar()
    # Quitar secciones de ruido (miramos toda la ruta: "Índice de contenido › ..." también cae)
    return [(ruta, c) for ruta, c in secciones if not any(r in sin_acentos(ruta) for r in SECCIONES_RUIDO)]


def chunks_por_secciones(doc: dict) -> list[dict]:
    manual = nombre_manual(doc["fuente"])
    salida = []
    if doc["fuente"].lower().endswith(".pdf"):
        # Partes alternas: ["", "1", texto_pág1, "2", texto_pág2, ...]
        partes = RE_PAGINA.split(doc["texto"])
        for num, cuerpo in zip(partes[1::2], partes[2::2]):
            for trozo in empacar_parrafos(cuerpo, config.CHUNK_MAX):
                salida.append({"texto": f"{manual} › pág. {num}\n\n{trozo}", "pagina": int(num)})
    else:
        for ruta, cuerpo in secciones_markdown(doc["texto"]):
            for trozo in empacar_parrafos(cuerpo, config.CHUNK_MAX):
                salida.append({"texto": f"{manual} › {ruta}\n\n{trozo}", "seccion": ruta})
    return salida


# ---------------------------------------------------------------- Común
def crear_chunks(documentos: list[dict], estrategia: str = config.CHUNKING) -> list[dict]:
    """Devuelve la lista de chunks con metadatos (fuente, chunk, tipo, pagina|seccion)."""
    partir = chunks_por_secciones if estrategia == "secciones" else chunks_por_tamano
    todos = []
    for doc in documentos:
        for i, c in enumerate(partir(doc)):
            todos.append({
                "id": f"{doc['fuente']}#{i}",   # identificador único, ej. manual_tickets.pdf#3
                "fuente": doc["fuente"],
                "chunk": i,
                "tipo": "pdf" if doc["fuente"].lower().endswith(".pdf") else "md",
                **c,
            })
    return todos


def describir(c: dict) -> str:
    """Texto corto para citar un chunk: 'archivo · pág. 4' o 'archivo · § sección'."""
    ubicacion = f"pág. {c['pagina']}" if "pagina" in c else f"§ {c['seccion']}"
    return f"{c['fuente']} · {ubicacion}"


if __name__ == "__main__":
    from collections import Counter

    docs = cargar_documentos()
    for estrategia in ("tamano", "secciones"):
        chunks = crear_chunks(docs, estrategia)
        largos = sorted(len(c["texto"]) for c in chunks)
        print(f"\n[{estrategia}] {len(chunks)} chunks · largo mediano {largos[len(largos) // 2]} · máximo {largos[-1]}")
        for fuente, n in Counter(c["fuente"] for c in chunks).items():
            print(f"  {fuente:<50} {n:>4}")

    # Ejemplo: los primeros chunks por sección, con su ruta al inicio
    print("\n" + "=" * 70)
    for c in crear_chunks(docs, "secciones")[:2]:
        print(f"--- {c['id']}  ({len(c['texto'])} caracteres) ---\n{c['texto']}\n")
