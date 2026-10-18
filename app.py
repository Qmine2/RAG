"""Interface web : streamlit run app.py"""
import streamlit as st

import rag

st.set_page_config(page_title="Assistant Télécom (RAG)", page_icon="📡")
st.title("📡 Assistant Support Télécom")
st.caption("RAG sur FAQ, tickets résolus et guide technique - propulsé par Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['meta']['ref']}** (score {s['score']})")
                    st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))

if q := st.chat_input("Posez votre question (FR / EN / AR)…"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours…"):
            try:
                out = rag.answer(q, history)
            except Exception as e:  # clé manquante, index absent, réseau…
                st.error(f"Erreur : {e}")
                st.stop()
        st.markdown(out["answer"])
        with st.expander("Sources"):
            for s in out["sources"]:
                st.markdown(f"**{s['meta']['ref']}** (score {s['score']})")
                st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
    )
