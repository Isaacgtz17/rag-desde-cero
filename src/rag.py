"""Paso 5 del RAG: pregunta -> recuperar chunks -> prompt con contexto -> Claude -> respuesta + fuentes.

Uso:  python src/rag.py "¿Cómo abro un ticket de soporte?"

Requiere la API key en un archivo .env en la raíz del proyecto:
    ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import sys

import anthropic
from dotenv import load_dotenv

import config
from buscar import buscar
from chunking import describir

load_dotenv(config.RAIZ / ".env")   # lee ANTHROPIC_API_KEY del archivo .env

# Instrucciones fijas para el modelo. Lo importante de un prompt RAG:
# 1) responder SOLO con el contexto, 2) admitir cuando no está, 3) citar fuentes.
SYSTEM = """Eres un asistente que responde preguntas sobre los manuales internos de la empresa.

Reglas:
- Responde SOLO con la información de los fragmentos de contexto que se te dan.
- Si la respuesta no está en el contexto, di: "No encontré esa información en los manuales." No inventes.
- Cita los fragmentos que usaste con su número entre corchetes, por ejemplo [1] o [2][3].
- Responde en español, de forma clara y breve. Si son pasos, usa una lista numerada."""


def armar_prompt(pregunta: str, fragmentos: list[dict]) -> str:
    """Junta los chunks recuperados en un bloque de contexto numerado + la pregunta."""
    # El metadato de sección/página también le da contexto al LLM sobre de dónde viene cada trozo
    contexto = "\n\n".join(
        f"[{i}] (fuente: {describir(f)})\n{f['texto']}"
        for i, f in enumerate(fragmentos, start=1)
    )
    return f"<contexto>\n{contexto}\n</contexto>\n\nPregunta: {pregunta}"


def llamar_claude(pregunta: str, fragmentos: list[dict]):
    """Arma el prompt y llama a Claude. Devuelve la respuesta completa de la API
    (el evaluador la necesita entera: texto, modelo, tokens usados, stop_reason)."""
    client = anthropic.Anthropic()   # toma la API key de la variable de entorno
    return client.beta.messages.create(
        model=config.MODELO_LLM,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},   # cuánto "piensa" el modelo; medium basta para preguntas de manual
        # Si el filtro de seguridad rechazara la petición, la API reintenta sola con otro modelo
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": armar_prompt(pregunta, fragmentos)}],
    )


def texto_de(respuesta) -> str:
    """Extrae el texto final de la respuesta de la API."""
    if respuesta.stop_reason == "refusal":
        return "(El modelo rechazó la petición.)"
    return "".join(b.text for b in respuesta.content if b.type == "text")


def responder(pregunta: str, fuente: str | None = None) -> tuple[str, list[dict]]:
    """Pipeline completo. `fuente` opcional: limitar la búsqueda a un manual (filtro por metadato)."""
    fragmentos = buscar(pregunta, fuente=fuente)
    return texto_de(llamar_claude(pregunta, fragmentos)), fragmentos


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Falta la API key: copia .env.example como .env y pon tu ANTHROPIC_API_KEY.")
    pregunta = " ".join(sys.argv[1:]) or "¿Cómo abro un ticket de soporte?"
    print(f"Pregunta: {pregunta}\n")
    texto, fragmentos = responder(pregunta)
    print(texto)
    print("\nFuentes:")
    for i, f in enumerate(fragmentos, start=1):
        print(f"  [{i}] {describir(f)}  (chunk {f['chunk']}, score {f['score']:.3f})")
