"""
Minimal-Test: funktioniert automatisches Function-Calling in diesem Setup
ueberhaupt, komplett unabhaengig von DataFrame/Closures/System-Instruction?

Run mit:  python chatbot_debug_repro.py
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]

MODEL = "gemini-3.1-flash-lite"  # bewusst NICHT die Lite-Variante, zum Testen


def add(a: int, b: int) -> int:
    """
    Adds two integers together.

    Args:
        a: The first number.
        b: The second number.
    """
    print(f"[DEBUG] add() wurde aufgerufen mit a={a}, b={b}")
    return a + b


client = genai.Client(api_key=api_key)
chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(tools=[add]),
)

response = chat.send_message("Was ist 7 plus 5? Nutze dafuer das verfuegbare Tool.")
print("Antwort:", response.text)