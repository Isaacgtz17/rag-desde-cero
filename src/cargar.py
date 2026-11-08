"""Paso 1 del RAG: cargar documentos y obtener texto limpio.

- Markdown: se lee tal cual (ya es texto).
- PDF: son imágenes, así que cada página pasa por OCR (Tesseract).
  El resultado se guarda en caché porque el OCR es lento (~2-4 s por página).

Uso para probar:  python src/cargar.py
"""
import os
import re

import pymupdf
import pytesseract
from PIL import Image

import config

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_EXE
os.environ["TESSDATA_PREFIX"] = str(config.TESSDATA_DIR)

# Líneas de ruido que el navegador agregó al "imprimir a PDF".
# Cada patrón se probó contra el texto real del OCR.
PATRONES_RUIDO = [
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2},\s*\d{1,2}:\d{2}\s*[AP]M"),  # "11/7/26, 10:50 PM Título de la página"
    re.compile(r"^https?://\S+"),                                     # URL del pie de página
] + [re.compile(p) for p in config.PATRONES_RUIDO_EXTRA]              # propios de tus documentos (config_local.py)


def limpiar_ocr(texto: str) -> str:
    """Quita encabezados/pies del navegador y líneas que son pura basura."""
    limpias = []
    for linea in texto.splitlines():
        linea = linea.strip()
        if any(p.search(linea) for p in PATRONES_RUIDO):
            continue
        # Basura típica de OCR sobre logos/iconos: líneas muy cortas sin palabras reales.
        # Ejemplos reales: "o)", "=> PE E. , Inicia Sesión _."  ->  pocas letras seguidas
        if linea and not re.search(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]{3,}", linea):
            continue
        limpias.append(linea)
    # Colapsar 3+ saltos de línea seguidos en uno doble (separador de párrafo)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(limpias)).strip()


SEPARADOR_PAGINA = "\n\f\n"   # \f = salto de página; no aparece en texto normal


def ocr_pdf(ruta_pdf) -> str:
    """OCR página por página. Devuelve el texto LIMPIO con marcas [Página N].

    La caché guarda el OCR CRUDO: así, si cambias las reglas de limpieza,
    no hay que repetir el OCR (que es lo lento).
    """
    cache = config.OCR_CACHE / (ruta_pdf.stem + ".txt")
    if cache.exists():                       # ya se procesó antes: no repetir OCR
        paginas_crudas = cache.read_text(encoding="utf-8").split(SEPARADOR_PAGINA)
    else:
        paginas_crudas = []
        doc = pymupdf.open(ruta_pdf)
        for i, pagina in enumerate(doc, start=1):
            print(f"  OCR {ruta_pdf.name} pág {i}/{doc.page_count}...")
            pix = pagina.get_pixmap(dpi=config.OCR_DPI)    # página -> imagen
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            paginas_crudas.append(pytesseract.image_to_string(img, lang="spa"))
        config.OCR_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(SEPARADOR_PAGINA.join(paginas_crudas), encoding="utf-8")

    # La limpieza se aplica siempre al cargar (barata y fácil de ajustar)
    return "\n\n".join(
        f"[Página {i}]\n{limpiar_ocr(t)}" for i, t in enumerate(paginas_crudas, start=1)
    )


def cargar_documentos() -> list[dict]:
    """Devuelve una lista de {'fuente': nombre_archivo, 'texto': ...}."""
    documentos = []
    for ruta in sorted(config.DATA_RAW.iterdir()):
        if ruta.suffix.lower() == ".md":
            texto = ruta.read_text(encoding="utf-8")
        elif ruta.suffix.lower() == ".pdf":
            texto = ocr_pdf(ruta)
        else:
            print(f"  (ignorado, formato no soportado: {ruta.name})")
            continue
        documentos.append({"fuente": ruta.name, "texto": texto})
    return documentos


if __name__ == "__main__":
    docs = cargar_documentos()
    print(f"\n{len(docs)} documentos cargados:\n")
    for d in docs:
        print(f"  {d['fuente']:<50} {len(d['texto']):>7,} caracteres")
