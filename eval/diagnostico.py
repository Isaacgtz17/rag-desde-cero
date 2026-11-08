"""Diagnóstico de recuperación (etapa 4): ¿en qué LUGAR queda el chunk correcto?

Cada caso dice qué textos DEBE contener el chunk correcto (no su id), así el
diagnóstico sigue sirviendo aunque cambie el chunking y los ids.

No llama al LLM: no gasta dinero. Mide solo la recuperación.

Los casos están en eval/casos_diagnostico.json (privado: depende de tus documentos).
Formato: ver eval/casos_diagnostico.ejemplo.json.

Uso:  python eval/diagnostico.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from buscar import MODOS, buscar

ARCHIVO_CASOS = Path(__file__).resolve().parent / "casos_diagnostico.json"
if not ARCHIVO_CASOS.exists():
    sys.exit(f"Falta {ARCHIVO_CASOS.name}: cópialo de casos_diagnostico.ejemplo.json y escribe casos de TUS documentos.")
# (pregunta, manual esperado, textos que el chunk correcto debe contener TODOS)
CASOS = [(c["pregunta"], c["fuente"], c["debe_contener"])
         for c in json.loads(ARCHIVO_CASOS.read_text(encoding="utf-8"))]


def lugar_correcto(resultados: list[dict], fuente: str, textos: list[str]):
    """Lugar (1, 2, ...) del primer chunk que cumple el caso, o None si no aparece."""
    for lugar, r in enumerate(resultados, start=1):
        if r["fuente"] == fuente and all(t in r["texto"] for t in textos):
            return lugar
    return None


if __name__ == "__main__":
    lugares = {m: [] for m in MODOS}
    tiempos = {m: 0.0 for m in MODOS}
    for pregunta, fuente, textos in CASOS:
        for modo in MODOS:
            t0 = time.time()
            # Pedimos muchos resultados para ver el lugar aunque quede fuera del top-5
            resultados = buscar(pregunta, k=config.CANDIDATOS * 2, modo=modo)
            tiempos[modo] += time.time() - t0
            lugares[modo].append(lugar_correcto(resultados, fuente, textos))

    ancho = 52
    print(f"\n{'Pregunta':<{ancho}}" + "".join(f"{m:>16}" for m in MODOS))
    print("-" * (ancho + 16 * len(MODOS)))
    for i, (pregunta, _, _) in enumerate(CASOS):
        celdas = ""
        for m in MODOS:
            l = lugares[m][i]
            marca = "✔" if l and l <= config.TOP_K else " "
            celdas += f"{(str(l) if l else '—') + ' ' + marca:>16}"
        print(f"{pregunta[:ancho - 2]:<{ancho}}{celdas}")
    print("-" * (ancho + 16 * len(MODOS)))
    print(f"{f'Aciertos en top-{config.TOP_K}':<{ancho}}"
          + "".join(f"{sum(1 for l in lugares[m] if l and l <= config.TOP_K):>14}/{len(CASOS)}" for m in MODOS))
    print(f"{'Tiempo promedio por pregunta (s)':<{ancho}}"
          + "".join(f"{tiempos[m] / len(CASOS):>16.2f}" for m in MODOS))
    print("\n— = el chunk correcto no está entre los candidatos (o no existe completo, ej. tabla partida)")
