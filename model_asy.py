import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("wyniki", exist_ok=True)

# --- MODEL REGRESJI: ILE ASOW ZROBI SERWUJACY ---
# To drugi typ problemu. Klasyfikacja odpowiada "kto wygra" (tak/nie),
# regresja odpowiada "ile" - czyli przewiduje liczbe.

df = pd.read_csv("wyniki/dane_czyste.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data")

# potrzebujemy wzrostu i rankingu obu zawodnikow
df = df.dropna(subset=["winner_ht", "loser_ht", "winner_rank", "loser_rank",
                       "winner_hand", "loser_hand"])
print("Meczów do modelu:", len(df))

# --- PRZYGOTOWANIE DANYCH ---
# Z jednego meczu robimy DWA wiersze: raz serwuje zwyciezca, raz przegrany.
# Dzieki temu model uczy sie serwisu, a nie tego kto wygral mecz.

wiersze = []

for i in range(len(df)):
    mecz = df.iloc[i]

    # wiersz 1: serwuje zwyciezca
    wiersze.append({
        "data": mecz["data"],
        "surface": mecz["surface"],
        "serwujacy": mecz["winner_name"],
        "przeciwnik": mecz["loser_name"],
        "wzrost_serwujacego": mecz["winner_ht"],
        "wzrost_przeciwnika": mecz["loser_ht"],
        "przeciwnik_leworeczny": 1 if mecz["loser_hand"] == "L" else 0,
        "ranking_przeciwnika": mecz["loser_rank"],
        "asy": mecz["asy_zwyciezca"]
    })

    # wiersz 2: serwuje przegrany
    wiersze.append({
        "data": mecz["data"],
        "surface": mecz["surface"],
        "serwujacy": mecz["loser_name"],
        "przeciwnik": mecz["winner_name"],
        "wzrost_serwujacego": mecz["loser_ht"],
        "wzrost_przeciwnika": mecz["winner_ht"],
        "przeciwnik_leworeczny": 1 if mecz["winner_hand"] == "L" else 0,
        "ranking_przeciwnika": mecz["winner_rank"],
        "asy": mecz["asy_przegrany"]
    })

dane = pd.DataFrame(wiersze)

# nawierzchnia jako kolumny 0/1
dane["twarda"] = (dane["surface"] == "Hard").astype(int)
dane["ziemia"] = (dane["surface"] == "Clay").astype(int)
dane["trawa"] = (dane["surface"] == "Grass").astype(int)

print("Wierszy do uczenia:", len(dane))

# --- PODZIAL PO CZASIE ---

granica = int(len(dane) * 0.8)
uczenie = dane.iloc[:granica]
test = dane.iloc[granica:]

# --- HISTORIA ZAWODNIKA JAKO CECHA ---
# Najmocniejszy czynnik to po prostu jak zawodnik serwuje na co dzien.
# UWAGA: srednie liczymy TYLKO z okresu uczenia. Gdybysmy policzyli je
# z calosci, model znalby przyszlosc - to wlasnie wyciek danych.

srednia_ogolna = uczenie["asy"].mean()

# ile asow zwykle robi serwujacy
serwis_gracza = uczenie.groupby("serwujacy")["asy"].mean()
# ile asow zwykle traci przeciwnik
straty_gracza = uczenie.groupby("przeciwnik")["asy"].mean()

# dopisujemy do obu zbiorow, nieznany zawodnik dostaje srednia ogolna
dane["serwis_historia"] = dane["serwujacy"].map(serwis_gracza).fillna(srednia_ogolna)
dane["straty_przeciwnika"] = dane["przeciwnik"].map(straty_gracza).fillna(srednia_ogolna)

# dzielimy jeszcze raz, juz z nowymi kolumnami
uczenie = dane.iloc[:granica]
test = dane.iloc[granica:]

cechy = ["serwis_historia", "straty_przeciwnika",
         "wzrost_serwujacego", "wzrost_przeciwnika", "przeciwnik_leworeczny",
         "ranking_przeciwnika", "twarda", "ziemia", "trawa"]

X_uczenie = uczenie[cechy]
y_uczenie = uczenie["asy"]
X_test = test[cechy]
y_test = test["asy"]

# --- TRENOWANIE ---

model = LinearRegression()
model.fit(X_uczenie, y_uczenie)

przewidywania = model.predict(X_test)

blad = mean_absolute_error(y_test, przewidywania)
r2 = r2_score(y_test, przewidywania)

print()
print("=== WYNIKI MODELU REGRESJI ===")
print("Sredni blad (MAE):", round(blad, 2), "asa na 100 punktów serwisowych")
print("R2:", round(r2, 3))

# punkt odniesienia: zgadywanie sredniej
srednia = y_uczenie.mean()
blad_sredniej = mean_absolute_error(y_test, [srednia] * len(y_test))
print("Dla porównania, zgadywanie średniej daje błąd:", round(blad_sredniej, 2))

print()
print("=== WAGI MODELU ===")
for j in range(len(cechy)):
    print("  ", cechy[j], "=", round(model.coef_[j], 4))

# --- WYKRES: PRZEWIDYWANE KONTRA PRAWDZIWE ---

plt.figure(figsize=(7, 7))
plt.scatter(y_test, przewidywania, alpha=0.2, s=10)
plt.plot([0, 40], [0, 40], color="red", linewidth=1)
plt.xlabel("Rzeczywiste asy na 100 punktów")
plt.ylabel("Przewidywane asy na 100 punktów")
plt.title("Model regresji - przewidywania kontra rzeczywistość")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("wyniki/wykres_model_asy.png", dpi=120)
plt.close()

# --- ZAPIS MODELU ---

# zapisujemy model razem ze srednimi zawodnikow, bo aplikacja ich potrzebuje
plik = open("wyniki/model_asy.pkl", "wb")
pickle.dump({
    "model": model,
    "cechy": cechy,
    "serwis_gracza": serwis_gracza.to_dict(),
    "straty_gracza": straty_gracza.to_dict(),
    "srednia_ogolna": srednia_ogolna
}, plik)
plik.close()

print()
print("Zapisano wyniki/model_asy.pkl oraz wyniki/wykres_model_asy.png")
