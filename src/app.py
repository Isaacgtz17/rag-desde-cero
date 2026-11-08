"""Interfaz web del RAG (etapa 6).

Uso:  streamlit run src/app.py      -> http://localhost:8501

Cómo funciona Streamlit (lo importante):
- Re-ejecuta ESTE script completo en cada interacción (clic, mensaje, voto).
- Lo que debe sobrevivir entre ejecuciones va en st.session_state (el historial).
- @st.cache_resource guarda objetos pesados (modelos) una sola vez por servidor.
"""
import json
import time
from datetime import datetime

import anthropic
import streamlit as st

import config
from buscar import coleccion, manuales, reranker
from chunking import describir
from embeddings import modelo
from rag import responder

FEEDBACK = config.RAIZ / "eval" / "feedback.jsonl"   # votos 👍/👎 -> futuras preguntas de evaluación
EJEMPLOS = config.PREGUNTAS_EJEMPLO   # botones de la pantalla inicial (personalizables en config_local.py)

st.set_page_config(page_title=config.NOMBRE_APP, page_icon="📘", layout="centered")


@st.cache_resource(show_spinner="Cargando modelos (solo la primera vez)...")
def precargar():
    """Carga embeddings, reranker y base vectorial al abrir, no en la primera pregunta."""
    # Asignado a "_" a propósito: Streamlit MUESTRA en pantalla cualquier expresión suelta ("magic")
    _ = (modelo(), reranker(), coleccion())
    return manuales()


def guardar_voto(i: int):
    """Se llama al hacer clic en 👍/👎. Guarda pregunta, respuesta, fuentes y voto."""
    item = st.session_state.historial[i]
    voto = st.session_state[f"voto_{i}"]   # 1 = 👍, 0 = 👎, None = quitó el voto
    if voto is None:
        return
    FEEDBACK.parent.mkdir(exist_ok=True)
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "voto": "bien" if voto == 1 else "mal",
            "pregunta": item["pregunta"],
            "respuesta": item["respuesta"],
            "fuentes": [f["id"] for f in item["fragmentos"]],
        }, ensure_ascii=False) + "\n")
    st.toast("¡Gracias! Voto guardado." if voto == 1 else "Anotado: lo usaremos para mejorar.")


def mostrar(i: int, item: dict, detalles: bool):
    """Dibuja una pregunta con su respuesta, fuentes y botones de voto."""
    with st.chat_message("user"):
        st.write(item["pregunta"])
    with st.chat_message("assistant"):
        st.markdown(item["respuesta"])
        with st.expander(f"📄 Fuentes ({len(item['fragmentos'])})"):
            for n, f in enumerate(item["fragmentos"], start=1):
                st.markdown(f"**[{n}]** {describir(f)}")
                if detalles:   # para diagnosticar: ver el texto exacto que recibió el LLM
                    st.caption(f"id `{f['id']}` · score `{f['score']:.3f}`")
                    st.text(f["texto"])
        col_voto, col_info = st.columns([1, 3])
        with col_voto:
            st.feedback("thumbs", key=f"voto_{i}", on_change=guardar_voto, args=(i,))
        with col_info:
            st.caption(f"⏱ {item['segundos']:.1f} s · {item['filtro'] or 'todos los manuales'}")


def preguntar(pregunta: str, fuente: str | None):
    t0 = time.time()
    try:
        with st.spinner("Buscando en los manuales y preguntando a Claude..."):
            respuesta, fragmentos = responder(pregunta, fuente=fuente)
    except anthropic.AuthenticationError:
        st.error("La API key no es válida. Revisa el archivo .env.")
        return
    except anthropic.APIError as e:
        st.error(f"Error al llamar a Claude: {e}")
        return
    st.session_state.historial.append({
        "pregunta": pregunta, "respuesta": respuesta, "fragmentos": fragmentos,
        "segundos": time.time() - t0, "filtro": fuente,
    })


# ------------------------------------------------------------------ Página
lista_manuales = precargar()
if "historial" not in st.session_state:
    st.session_state.historial = []

with st.sidebar:
    st.header("⚙️ Opciones")
    opcion = st.selectbox("Buscar en", ["Todos los manuales"] + lista_manuales)
    fuente = None if opcion == "Todos los manuales" else opcion
    detalles = st.toggle("Mostrar detalles técnicos", help="Texto y score de cada chunk recuperado")
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.historial = []
        st.rerun()
    st.divider()
    st.caption(f"Búsqueda: {config.MODO_BUSQUEDA} · top-{config.TOP_K}\n\nLLM: {config.MODELO_LLM}")

st.title(f"📘 {config.NOMBRE_APP}")
st.caption("Respuestas basadas solo en los manuales internos, con sus fuentes. Si no está en los manuales, te lo dirá.")

# Pantalla vacía: preguntas de ejemplo como botones
ejemplo = None
if not st.session_state.historial:
    st.write("**Prueba con:**")
    cols = st.columns(2)
    for n, texto in enumerate(EJEMPLOS):
        if cols[n % 2].button(texto, use_container_width=True):
            ejemplo = texto

for i, item in enumerate(st.session_state.historial):
    mostrar(i, item, detalles)

pregunta = st.chat_input("Escribe tu pregunta sobre los manuales...") or ejemplo
if pregunta:
    preguntar(pregunta, fuente)
    st.rerun()   # redibujar con la nueva respuesta (y sus botones de voto)
