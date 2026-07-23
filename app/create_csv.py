import numpy as np
import pandas as pd

#small dataset

"""
data = {
    "kunden_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "alter": [24, 45, 31, 52, 19, 38, 61, 29, 41, 27, 35, 50, 22, 48, 33],
    "geschlecht": ["W", "M", "W", "M", "W", "M", "W", "M", "W", "M", "W", "M", "W", "M", "W"],
    "einkommen_k": [32.5, 58.0, 42.0, 75.0, 18.5, 49.0, 62.0, 39.0, 51.5, 35.0, 47.0, 68.0, 21.0, 55.0, 44.0],
    "gekaufte_buecher": [3, 7, 2, 12, 1, 5, 8, 4, 6, 2, 5, 9, 3, 6, 4],
    "ausgaben_eur": [45.90, 120.50, 29.90, 210.00, 14.99, 85.00, 145.20, 62.00, 98.50, 34.50, 79.90, 160.00, 42.00, 105.00, 68.50],
    "bevorzugtes_genre": ["Fiction", "Sachbuch", "Thriller", "Sachbuch", "Fantasy", "Thriller", "Romance", "Fiction", "Fantasy", "Thriller", "Fiction", "Sachbuch", "Fantasy", "Thriller", "Romance"],
    "bewertung": [4.5, 4.0, 3.5, 5.0, 2.0, 4.0, 4.8, 3.0, 4.2, 3.8, 4.1, 4.7, 3.0, 4.3, 3.9],
    "treue_mitglied": ["Ja", "Ja", "Nein", "Ja", "Nein", "Ja", "Ja", "Nein", "Ja", "Nein", "Ja", "Ja", "Nein", "Ja", "Nein"]
}

df = pd.DataFrame(data)
df.to_csv("kunden_buecher.csv", index=False)
print("Datei 'kunden_buecher.csv' wurde erfolgreich erstellt!")
"""


#2 big dataset

# Anzahl der Zeilen festlegen (z.B. 50.000 für einen soliden Stresstest)
num_rows = 50000
np.random.seed(42)

data = {
    "User_ID": np.arange(1000, 1000 + num_rows),
    "Alter": np.random.randint(18, 70, size=num_rows),
    "Einkommen_EUR": np.random.normal(loc=3500, scale=1200, size=num_rows).round(
        2
    ),
    "Kategorie": np.random.choice(
        ["Elektronik", "Kleidung", "Haushalt", "Bücher"], size=num_rows
    ),
    "Zufriedenheit_Score": np.random.uniform(1.0, 5.0, size=num_rows).round(1),
    "Kaufentscheidung": np.random.choice([0, 1], size=num_rows, p=[0.7, 0.3]),
}

df = pd.DataFrame(data)

# Streue ein paar zufällige NaN-Werte ein, um die Fehlerbehandlung deiner API zu testen
df.loc[df.sample(frac=0.05).index, "Einkommen_EUR"] = np.nan
df.loc[df.sample(frac=0.02).index, "Kategorie"] = None

# Als CSV speichern
df.to_csv("big_test_statistik_dataset.csv", index=False)
print("CSV erfolgreich erstellt mit", len(df), "Zeilen!")