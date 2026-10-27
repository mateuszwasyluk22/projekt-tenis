import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# wczytujemy dane z policzonym elo
df = pd.read_csv("wyniki/dane_z_elo.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data")

# potrzebujemy rankingu ATP wiec usuwamy mecze bez niego
df = df.dropna(subset=["winner_rank", "loser_rank", "winner_ht", "loser_ht"])
print("Meczów do modelu:", len(df))

# --- PRZYGOTOWANIE DANYCH ---
# W danych zwyciezca jest zawsze pierwszy, wiec model mialby gotowa odpowiedz.
# Dlatego co drugi mecz odwracamy: raz pierwszy jest zwyciezca, raz przegrany.

wiersze = []

for i in range(len(df)):
    mecz = df.iloc[i]

    if i % 2 == 0:
        # zawodnik A to zwyciezca, wynik 1
        wynik = 1
        elo_a = mecz["elo_zwyciezca"]
        elo_b = mecz["elo_przegrany"]
        elo_naw_a = mecz["elo_naw_zwyciezca"]
        elo_naw_b = mecz["elo_naw_przegrany"]
        rank_a = mecz["winner_rank"]
        rank_b = mecz["loser_rank"]
        wzrost_a = mecz["winner_ht"]
        wzrost_b = mecz["loser_ht"]
    else:
        # zawodnik A to przegrany, wynik 0
        wynik = 0
        elo_a = mecz["elo_przegrany"]
        elo_b = mecz["elo_zwyciezca"]
        elo_naw_a = mecz["elo_naw_przegrany"]
        elo_naw_b = mecz["elo_naw_zwyciezca"]
        rank_a = mecz["loser_rank"]
        rank_b = mecz["winner_rank"]
        wzrost_a = mecz["loser_ht"]
        wzrost_b = mecz["winner_ht"]

    wiersze.append({
        "data": mecz["data"],
        "surface": mecz["surface"],
        "roznica_elo": elo_a - elo_b,
        "roznica_elo_naw": elo_naw_a - elo_naw_b,
        "roznica_rank": rank_b - rank_a,
        "roznica_wzrost": wzrost_a - wzrost_b,
        "wynik": wynik
    })

dane = pd.DataFrame(wiersze)

# nawierzchnia jako trzy kolumny 0/1 - model rozumie tylko liczby
dane["twarda"] = (dane["surface"] == "Hard").astype(int)
dane["ziemia"] = (dane["surface"] == "Clay").astype(int)
dane["trawa"] = (dane["surface"] == "Grass").astype(int)

# --- PODZIAL NA UCZENIE I TEST ---
# Dzielimy po czasie, nie losowo. Model uczy sie na starszych meczach,
# a sprawdzamy go na najnowszych - tak jak bylo by w rzeczywistosci.

granica = int(len(dane) * 0.8)
uczenie = dane.iloc[:granica]
test = dane.iloc[granica:]

print("Mecze do uczenia:", len(uczenie))
print("Mecze do testu:", len(test))
print("Test obejmuje okres od", test["data"].min().date(), "do", test["data"].max().date())

# --- TRZY WARIANTY MODELU ---

warianty = {
    "Tylko ranking ATP": ["roznica_rank"],
    "Tylko Elo": ["roznica_elo"],
    "Wszystko razem": ["roznica_elo", "roznica_elo_naw", "roznica_rank",
                       "roznica_wzrost", "twarda", "ziemia", "trawa"]
}

wyniki = {}

for nazwa in warianty:
    cechy = warianty[nazwa]

    X_uczenie = uczenie[cechy]
    y_uczenie = uczenie["wynik"]
    X_test = test[cechy]
    y_test = test["wynik"]

    # trenujemy regresje logistyczna
    model = LogisticRegression(max_iter=1000)
    model.fit(X_uczenie, y_uczenie)

    # sprawdzamy na meczach ktorych model nie widzial
    przewidywania = model.predict(X_test)
    dokladnosc = accuracy_score(y_test, przewidywania)
    wyniki[nazwa] = dokladnosc

    print()
    print("=== WARIANT: " + nazwa + " ===")
    print("Dokładność:", round(dokladnosc * 100, 2), "%")
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, przewidywania))

    # ostatni wariant pokazujemy dokladniej
    if nazwa == "Wszystko razem":
        print("Wagi modelu:")
        for j in range(len(cechy)):
            print("  ", cechy[j], "=", round(model.coef_[0][j], 4))
        najlepszy_model = model
        najlepsze_cechy = cechy

# --- PODSUMOWANIE ---

print()
print("=== PORÓWNANIE WARIANTÓW ===")
for nazwa in wyniki:
    print(nazwa, "->", round(wyniki[nazwa] * 100, 2), "%")

# --- PRZYKLADOWE PROGNOZY ---
# pokazujemy szanse na wygrana dla kilku meczow z testu

szanse = najlepszy_model.predict_proba(test[najlepsze_cechy])[:, 1]
podglad = test.copy()
podglad["szansa_A"] = (szanse * 100).round(1)
podglad["roznica_elo"] = podglad["roznica_elo"].round(1)

print()
print("=== PRZYKŁADOWE PROGNOZY (szansa zawodnika A w procentach) ===")
print(podglad[["data", "surface", "roznica_elo", "szansa_A", "wynik"]].head(10))

# zapisujemy wytrenowany model, zeby aplikacja mogla go uzyc
plik = open("wyniki/model_wygrana.pkl", "wb")
pickle.dump({"model": najlepszy_model, "cechy": najlepsze_cechy}, plik)
plik.close()

print()
print("Zapisano wyniki/model_wygrana.pkl")

# zapisujemy wyniki do pliku tekstowego
plik = open("wyniki/wyniki_modelu.txt", "w")
for nazwa in wyniki:
    plik.write(nazwa + " -> " + str(round(wyniki[nazwa] * 100, 2)) + " %\n")
plik.close()

print()
print("Zapisano wyniki/wyniki_modelu.txt")
