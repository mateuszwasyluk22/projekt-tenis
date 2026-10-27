import pandas as pd
import matplotlib.pyplot as plt
import os

# folder na wyniki
os.makedirs("wyniki", exist_ok=True)

# wczytujemy oczyszczone dane z poprzedniego kroku
df = pd.read_csv("wyniki/dane_czyste.csv")
print("Wczytano meczów:", len(df))

# --- WYKRES 1: LICZBA MECZOW WEDLUG NAWIERZCHNI ---

licznik = df["surface"].value_counts()

plt.figure(figsize=(8, 5))
licznik.plot(kind="bar", color="steelblue")
plt.title("Liczba meczów według nawierzchni")
plt.xlabel("Nawierzchnia")
plt.ylabel("Liczba meczów")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("wyniki/wykres_nawierzchnie.png", dpi=120)
plt.close()

print()
print("=== LICZBA MECZÓW WEDŁUG NAWIERZCHNI ===")
print(licznik)

# --- WYKRES 2: SREDNIA LICZBA ASOW WEDLUG NAWIERZCHNI ---

# liczymy srednia dla zwyciezcow i przegranych osobno
asy = df.groupby("surface")[["asy_zwyciezca", "asy_przegrany"]].mean()

plt.figure(figsize=(8, 5))
asy.plot(kind="bar", ax=plt.gca())
plt.title("Średnia liczba asów na 100 punktów serwisowych")
plt.xlabel("Nawierzchnia")
plt.ylabel("Asy na 100 punktów")
plt.xticks(rotation=0)
plt.legend(["Zwycięzca", "Przegrany"])
plt.tight_layout()
plt.savefig("wyniki/wykres_asy.png", dpi=120)
plt.close()

print()
print("=== ŚREDNIE ASY WEDŁUG NAWIERZCHNI ===")
print(asy.round(2))

# --- WYKRES 3: TOP 10 ZAWODNIKOW ---

top10 = df["winner_name"].value_counts().head(10)

plt.figure(figsize=(9, 6))
top10.sort_values().plot(kind="barh", color="seagreen")
plt.title("TOP 10 zawodników według liczby wygranych")
plt.xlabel("Wygrane mecze")
plt.tight_layout()
plt.savefig("wyniki/wykres_ranking.png", dpi=120)
plt.close()

print()
print("=== TOP 10 ZAWODNIKÓW ===")
print(top10)

# --- WYKRES 4: KORELACJE MIEDZY STATYSTYKAMI ---

# wybieramy kolumny liczbowe do korelacji
do_korelacji = ["asy_zwyciezca", "bledy_zwyciezca", "pierwszy_zwyciezca",
                "asy_przegrany", "bledy_przegrany", "pierwszy_przegrany"]

korelacje = df[do_korelacji].corr()

plt.figure(figsize=(9, 7))
plt.imshow(korelacje, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Korelacja")
plt.xticks(range(len(do_korelacji)), do_korelacji, rotation=45, ha="right")
plt.yticks(range(len(do_korelacji)), do_korelacji)

# wpisujemy liczby w kratki
for i in range(len(do_korelacji)):
    for j in range(len(do_korelacji)):
        wartosc = korelacje.iloc[i, j]
        plt.text(j, i, round(wartosc, 2), ha="center", va="center", fontsize=9)

plt.title("Korelacje między statystykami")
plt.tight_layout()
plt.savefig("wyniki/wykres_korelacje.png", dpi=120)
plt.close()

print()
print("=== KORELACJE ===")
print(korelacje.round(2))

# --- WYKRES 5: ASY PRZECIWKO LEWO- I PRAWORECZNYM ---

# interesuje nas reka przegranego, czyli tego ktory odbiera serwis
reka = df.groupby(["surface", "loser_hand"])["asy_zwyciezca"].mean().unstack()

# zostawiamy tylko lewa i prawa reke
kolumny = []
for k in ["L", "R"]:
    if k in reka.columns:
        kolumny.append(k)
reka = reka[kolumny]

plt.figure(figsize=(8, 5))
reka.plot(kind="bar", ax=plt.gca())
plt.title("Asy serwującego według ręki przeciwnika")
plt.xlabel("Nawierzchnia")
plt.ylabel("Asy na 100 punktów")
plt.xticks(rotation=0)
plt.legend(title="Ręka przeciwnika")
plt.tight_layout()
plt.savefig("wyniki/wykres_reka.png", dpi=120)
plt.close()

print()
print("=== ASY WEDŁUG RĘKI PRZECIWNIKA (L = leworęczny, R = praworęczny) ===")
print(reka.round(2))

print()
print("Zapisano 5 wykresów PNG")
