"""
Streamlit entry point - this is the file you actually run to start the app.

Run with (from the project root):
    streamlit run streamlit_app.py
"""

import logging
import streamlit as st
from app.config import get_secret
from app.data_handling import load_dataframe, numeric_columns
from app.chatbot import StatisticsChatbot

# Konsolen-Logging einrichten: technische Details landen hier, nicht im
# Chat-Fenster (siehe die try/except-Bloecke weiter unten).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Statistik Tool", page_icon="📊", layout="wide")


def _check_password() -> bool:
    """
    Optionaler Passwortschutz: nur aktiv, wenn ein APP_PASSWORD gesetzt
    ist (lokal in .env, nach Deployment in st.secrets). Ohne gesetztes
    Passwort bleibt die App wie bisher frei zugaenglich - so bricht
    nichts an deinem bisherigen lokalen Ablauf.
    """
    required_password = get_secret("APP_PASSWORD")
    if not required_password:
        return True  # kein Passwort konfiguriert -> kein Schutz noetig

    if st.session_state.get("authenticated"):
        return True

    entered = st.text_input("Passwort", type="password")
    if entered:
        if entered == required_password:
            st.session_state["authenticated"] = True
            st.rerun()  # Seite neu laden, jetzt als "authentifiziert"
        else:
            st.error("Falsches Passwort.")
    return False


if not _check_password():
    st.stop()

st.title("📊 Statistik Tool")
st.write("Lade eine CSV- oder Excel-Datei hoch und sprich mit dem Chatbot ueber deine Daten.")

uploaded_file = st.file_uploader("Datei auswaehlen", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = load_dataframe(uploaded_file)
    except ValueError as error:
        # load_dataframe() wirft ausschliesslich kontrollierte, bereits
        # auf Deutsch formulierte Nutzer-Fehlermeldungen - die zeigen wir
        # direkt an, das ist kein Sicherheitsrisiko.
        st.error(str(error))
        st.stop()

    st.success(f"'{uploaded_file.name}' geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")

    with st.expander("Vorschau & Spaltenuebersicht"):
        st.dataframe(df.head(20))
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("Datentyp pro Spalte:")
            st.write(df.dtypes.astype(str).rename("Typ"))
        with col_right:
            st.write("Numerische Spalten:")
            st.write(numeric_columns(df))

    if st.session_state.get("loaded_filename") != uploaded_file.name:
        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            st.error("Kein GEMINI_API_KEY gefunden (weder in .env noch in st.secrets).")
            st.stop()
        st.session_state["chatbot"] = StatisticsChatbot(df, api_key=api_key)
        st.session_state["messages"] = []
        st.session_state["loaded_filename"] = uploaded_file.name

    st.subheader("Chat")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Frag etwas zu deinen Daten...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Denke nach..."):
                try:
                    antwort = st.session_state["chatbot"].ask(user_input)
                except Exception:
                    # Volle Details landen im Server-Log (fuer dich als
                    # Entwickler), der Nutzer sieht nur eine generische,
                    # unverfaengliche Meldung - keine internen Fehler-
                    # details, Bibliotheks-Interna o.ae. im Chat.
                    logger.exception("Fehler beim Chatbot-Aufruf")
                    antwort = (
                        "Es gab ein technisches Problem bei der Anfrage. "
                        "Bitte versuch es gleich nochmal."
                    )
            st.write(antwort)
        st.session_state["messages"].append({"role": "assistant", "content": antwort})