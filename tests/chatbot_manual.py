"""
Manual, interactive check that the real Gemini API + our tool works
end-to-end. This is intentionally NOT a pytest test: it needs a real API
key and an actual network call, and the model's exact wording will vary
between runs - none of that is something an automated test should depend
on. Run it directly instead:

    python manual_chatbot_test.py
"""

import os
from dotenv import load_dotenv
import pandas as pd
from app.chatbot import StatisticsChatbot

load_dotenv()  # liest GEMINI_API_KEY aus der .env-Datei im Projekt-Root
api_key = os.environ["GEMINI_API_KEY"]

# Kleiner Beispiel-Datensatz zum Testen - spaeter kommt der echte
# hochgeladene Datensatz aus Streamlit hier rein (Phase 4)
sample_data = pd.DataFrame({
    "Zahlen": [23, 45, 31, 29, 52, 38, 41, 27],
})

bot = StatisticsChatbot(sample_data, api_key=api_key)

antwort = bot.ask("Berechne die deskriptive Statistik fuer die Spalte 'Zahlen'.")
print(antwort)
