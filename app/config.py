"""
Central place for reading configuration and secrets (API keys, optional
app password). Tries Streamlit's secrets store first (st.secrets - this
is what Streamlit Community Cloud uses for deployed apps), then falls
back to environment variables, which are populated locally from a .env
file via python-dotenv. This means the exact same code works both on
your machine and after deployment, without any changes.
"""

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # liest .env im Projekt-Root, falls vorhanden (nur lokal relevant)


def get_secret(key: str) -> str | None:
    """
    Looks up a single config/secret value by name.

    Args:
        key: Name of the secret, e.g. "GEMINI_API_KEY".

    Returns:
        The value if found (from st.secrets or an environment variable),
        otherwise None.
    """
    try:
        # st.secrets wirft eine Exception, wenn ueberhaupt keine
        # secrets.toml existiert (z.B. rein lokal ohne eine solche Datei) -
        # das ist kein Fehlerfall fuer uns, wir fallen dann einfach auf
        # Umgebungsvariablen zurueck.
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)