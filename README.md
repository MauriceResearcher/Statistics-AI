# 📊 LLM Statistics Assistant & Analyzer

Ein interaktives Streamlit-Tool zur automatisierten statistischen Datenanalyse von CSV- und Excel-Dateien. Das Projekt kombiniert die **Google Gemini API** (Function Calling) mit Python-Bibliotheken (**SciPy, Pandas, NumPy**), um statistische Auswertungen in natürlicher Sprache durchzuführen.

Das Modell erhält **niemals die eigentlichen Daten**, sondern ruft lokal definierte Python-Funktionen auf. Die Berechnungen finden vollständig lokal statt und werden für den Nutzer verständlich auf Deutsch interpretiert.

---

## 🚀 Features & Funktionsweise

* **Sicherer Datei-Upload:** Unterstützung für `.csv`, `.xlsx` und `.xls` mit automatischer Validierung, Typprüfung und Größenbeschränkung (max. 20 MB).
* **Datenschutz durch Tool Calling:** Gemini sieht nur Spaltennamen und Datentypen. Sämtliche statistische Berechnungen finden lokal auf deinem Server/Rechner statt.
* **Deskriptive Statistik:** Mittelwert, Median, Modus, Standardabweichung, Quartilsabstände und Spannweite auf Knopfdruck.
* **Inferenzstatistik (Hypothesentests):**
  * Einstichproben-t-Test & Zweistichproben-t-Test (Student's & Welch's)
  * Paarweiser t-Test (Vorher-/Nachher-Vergleiche)
  * Chi-Quadrat-Unabhängigkeitstest für kategoriale Variablen
* **Regressions- & Korrelationsanalysen:** Pearson- & Spearman-Korrelation, einfache und multiple lineare Regression.
* **Integrierter Passwortschutz:** Optionale Authentifizierung für den Zugriff auf die App.

---

## 🛠️ Tech Stack

* **Sprache & Package Manager:** Python 3.10+, `uv`
* **Frontend & UI:** Streamlit
* **LLM & Tool Binding:** Google GenAI SDK (`google-genai`), Gemini 3.1 Flash Lite
* **Data Science & Statistik:** Pandas, NumPy, SciPy

---

## 📂 Projektstruktur

```text
.
├── app/
│   ├── statistics/         # Reine Statistik-Module (SciPy/NumPy)
│   │   ├── descriptive.py  # Deskriptive Kennzahlen
│   │   ├── inferential.py  # Hypothesentests (t-Tests, Chi-Quadrat)
│   │   ├── regression.py   # Korrelationen & Regressionsmodelle
│   │   └── _utils.py       # Hilfsfunktionen & p-Wert-Interpretation
│   ├── chatbot.py          # Gemini SDK Integration & Tool-Definitionen
│   ├── data_handling.py    # File-Upload Parsing & Validierung
│   └── config.py           # Konfigurations- & Secrets-Management
├── streamlit_app.py        # Hauptprogramm & Streamlit-UI
├── pyproject.toml          # uv Projektkonfiguration & Abhängigkeiten
└── .env                    # Lokale Umgebungsvariablen (API-Keys)
```

## Quickstart

### 1. Repository klonen & Umgebung einrichten

```text
git clone [https://github.com/DEIN_USERNAME/statistik-tool.git](https://github.com/DEIN_USERNAME/statistik-tool.git)
cd statistik-tool
uv sync
```
### 2. Umgebungsvariablen konfigurieren
Erstelle eine .env-Datei im Hauptverzeichnis:
```text
GEMINI_API_KEY=dein_gemini_api_key_hier
APP_PASSWORD=optionales_passwort_fuer_app_zugriff
```

### 3. Anwendung starten
uv run streamlit run streamlit_app.py

## 📌 Roadmap & Geplante Updates

- [ ] **Datenvisualisierung:**
  - Automatische Generierung von Diagrammen (Boxplots, Streudiagramme, Histograms) via Plotly/Seaborn im Chat.**
- [ ] **Erweiterte Testverfahren:**
  - Integration von ANOVA, Mann-Whitney-U-Test und Logistischer Regression.**
- [ ] **Rollen- & Rechtekonzept:**
  - Unterscheidung zwischen User- und Admin-Tools für erweiterte Systemfunktionen.
- [ ] **Export-Funktion:**
  - PDF-Berichtgenerierung für durchgeführte Analysen.

---

