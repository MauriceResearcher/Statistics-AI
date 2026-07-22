"""
Streamlit entry point - this is the file you actually run to start the app.

Run with (from the project root):
    streamlit run streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from app.data_handling import load_dataframe, numeric_columns
from app.chatbot import StatisticsChatbot

load_dotenv()  # liest GEMINI_API_KEY aus der .env-Datei im Projekt-Root

st.set_page_config(page_title="Statistik Tool", page_icon="📊", layout="wide")
st.title("📊 Statistik Tool")
st.write("Lade eine CSV- oder Excel-Datei hoch und sprich mit dem Chatbot ueber deine Daten.")

uploaded_file = st.file_uploader("Datei auswaehlen", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = load_dataframe(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        # st.stop() beendet die Skriptausfuehrung an dieser Stelle sofort -
        # verhindert, dass der Rest des Skripts mit ungueltigen/fehlenden
        # Daten trotzdem weiterlaeuft.
        st.stop()

    st.success(f"'{uploaded_file.name}' geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")

    with st.expander("Vorschau & Spaltenuebersicht"):
        st.dataframe(df.head(20))
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("Datentyp pro Spalte:")
            # .astype(str) wandelt die numpy-dtype-Objekte (z.B. dtype('int64'))
            # in normale Text-Werte um - nur die sind fuer Streamlit/Arrow
            # problemlos darstellbar.
            st.write(df.dtypes.astype(str).rename("Typ"))
        with col_right:
            st.write("Numerische Spalten:")
            st.write(numeric_columns(df))

    # --- Session State ---
    # Wir erkennen einen neuen Upload daran, dass sich der Dateiname
    # geaendert hat, und bauen dann einen frischen Chatbot (mit Tools fuer
    # genau diese Datei) sowie eine leere Nachrichten-Historie auf. Ohne
    # diese Pruefung wuerden wir bei JEDEM Rerun (z.B. bei jeder neuen
    # Chat-Nachricht) einen komplett neuen Chatbot erzeugen und die
    # gesamte bisherige Unterhaltung verlieren.
    if st.session_state.get("loaded_filename") != uploaded_file.name:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Kein GEMINI_API_KEY in der .env-Datei gefunden.")
            st.stop()
        st.session_state["chatbot"] = StatisticsChatbot(df, api_key=api_key)
        st.session_state["messages"] = []
        st.session_state["loaded_filename"] = uploaded_file.name

    st.subheader("Chat")

    # Bisherige Nachrichten anzeigen - bei jedem Rerun neu gezeichnet,
    # die Inhalte selbst kommen aber aus der ueber Reruns hinweg
    # gespeicherten Historie in session_state.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # st.chat_input zeigt ein Eingabefeld am unteren Bildschirmrand.
    # Solange nichts eingegeben wurde, ist der Rueckgabewert None; sobald
    # der Nutzer abschickt, enthaelt er den eingegebenen Text, und das
    # Skript laeuft danach (wie immer bei Streamlit) von vorne durch.
    user_input = st.chat_input("Frag etwas zu deinen Daten...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                try:
                    antwort = st.session_state["chatbot"].ask(user_input)
                except Exception as error:
                    antwort = f"Es gab ein technisches Problem bei der Anfrage an den Chatbot: {error}"
            st.write(antwort)
        st.session_state["messages"].append({"role": "assistant", "content": antwort})
