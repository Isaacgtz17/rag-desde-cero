"""Reescritura de la pregunta (multi-query), mejora sugerida por la evaluación de la etapa 5.

Problema medido: los 3 fallos de la evaluación fueron por VOCABULARIO. El usuario dice
"prospecto" y el manual "lead"; el usuario dice "mesa de ayuda" y el manual "Centro de Soporte".

Solución: antes de buscar, un modelo barato y rápido (Haiku) genera 3 variantes de la pregunta
con términos que probablemente use un manual. Se busca con TODAS (la original + variantes),
se fusionan los resultados con RRF y el reranker ordena contra la pregunta ORIGINAL.

Si la llamada falla (sin internet, sin key, rechazo), se sigue solo con la pregunta original:
la reescritura es una mejora opcional, nunca debe tumbar la búsqueda.
"""
import json
from functools import lru_cache

import anthropic
from dotenv import load_dotenv

import config

load_dotenv(config.RAIZ / ".env")

# Solo describe el corpus (qué manuales hay y de qué tratan). NO incluye términos sacados
# de los casos que fallaron en la evaluación: eso sería sobreajustar al examen.
SYSTEM = f"""Reformulas preguntas de empleados para buscarlas en los manuales de usuario de la empresa.

Manuales disponibles:
{config.DESCRIPCION_CORPUS}

Escribe 3 versiones de la pregunta, en español, que busquen lo mismo con otras palabras:
usa términos formales o técnicos que usaría un manual de software, sinónimos de lo coloquial
y, si aplica, el nombre del sistema. No respondas la pregunta."""

ESQUEMA = {
    "type": "object",
    "properties": {"variantes": {"type": "array", "items": {"type": "string"}}},
    "required": ["variantes"],
    "additionalProperties": False,
}


@lru_cache(maxsize=256)   # misma pregunta -> no volver a pagar (la interfaz la pide dos veces)
def reescribir(pregunta: str) -> tuple[str, ...]:
    """Devuelve hasta 3 variantes de la pregunta (tupla vacía si algo falla)."""
    try:
        r = anthropic.Anthropic().messages.create(
            model=config.MODELO_REESCRITURA,
            max_tokens=1000,
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": ESQUEMA}},
            messages=[{"role": "user", "content": pregunta}],
        )
        if r.stop_reason != "end_turn":
            return ()
        texto = "".join(b.text for b in r.content if b.type == "text")
        return tuple(json.loads(texto)["variantes"][:3])
    except (anthropic.APIError, json.JSONDecodeError, KeyError) as e:
        print(f"(reescritura omitida: {e})")
        return ()


if __name__ == "__main__":
    import sys
    for v in reescribir(" ".join(sys.argv[1:]) or "¿cómo le pongo prioridad a un prospecto?"):
        print("-", v)
