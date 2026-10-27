import pandas as pd
import matplotlib.pyplot as plt
import os

# folder na wyniki
os.makedirs("wyniki", exist_ok=True)

# wczytujemy oczyszczone dane
df = pd.read_csv("wyniki/dane_czyste.csv")

# upewniamy sie ze mecze sa po kolei w czasie
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data")
print("Meczów do policzenia:", len(df))

# kazdy zawodnik startuje z 1500 punktami
START = 1500

# K mowi jak mocno jeden mecz zmienia ranking
K = 32

# slowniki trzymajace aktualne elo
elo = {}
elo_naw = {}

# tu zapiszemy elo PRZED kazdym meczem
elo_zw_lista = []
elo_prz_lista = []
elo_naw_zw_lista = []
elo_naw_prz_lista = []

# tu zapiszemy historie do wykresu
historia = []

for i in range(len(df)):
    mecz = df.iloc[i]

    zwyciezca = mecz["winner_name"]
    przegrany = mecz["loser_name"]
    nawierzchnia = mecz["surface"]

    # klucze dla elo liczonego osobno na kazdej nawierzchni
    klucz_zw = str(zwyciezca) + "_" + str(nawierzchnia)
    klucz_prz = str(przegrany) + "_" + str(nawierzchnia)

    # pobieramy aktualne elo, nowy zawodnik dostaje 1500
    elo_zw = elo.get(zwyciezca, START)
    elo_prz = elo.get(przegrany, START)
    elo_naw_zw = elo_naw.get(klucz_zw, START)
    elo_naw_prz = elo_naw.get(klucz_prz, START)

    # zapisujemy stan SPRZED meczu - model nie moze znac wyniku
    elo_zw_lista.append(elo_zw)
    elo_prz_lista.append(elo_prz)
    elo_naw_zw_lista.append(elo_naw_zw)
    elo_naw_prz_lista.append(elo_naw_prz)

    # szansa zwyciezcy wedlug elo przed meczem
    oczekiwana = 1 / (1 + 10 ** ((elo_prz - elo_zw) / 400))

    # zwyciezca zyskuje tyle ile przegrany traci
    zmiana = K * (1 - oczekiwana)
    elo[zwyciezca] = elo_zw + zmiana
    elo[przegrany] = elo_prz - zmiana

    # to samo osobno dla nawierzchni
    oczekiwana_naw = 1 / (1 + 10 ** ((elo_naw_prz - elo_naw_zw) / 400))
    zmiana_naw = K * (1 - oczekiwana_naw)
    elo_naw[klucz_zw] = elo_naw_zw + zmiana_naw
    elo_naw[klucz_prz] = elo_naw_prz - zmiana_naw

    # zapisujemy do historii zeby narysowac wykres formy
    historia.append({"data": mecz["data"], "gracz": zwyciezca, "elo": elo[zwyciezca]})
    historia.append({"data": mecz["data"], "gracz": przegrany, "elo": elo[przegrany]})

# dopisujemy kolumny z elo sprzed meczu do tabeli
df["elo_zwyciezca"] = elo_zw_lista
df["elo_przegrany"] = elo_prz_lista
df["elo_naw_zwyciezca"] = elo_naw_zw_lista
df["elo_naw_przegrany"] = elo_naw_prz_lista

# --- RANKING KONCOWY ---

ranking = pd.Series(elo).sort_values(ascending=False)

print()
print("=== TOP 15 – RANKING ELO NA KONIEC ===")
print(ranking.head(15).round(0))

# --- RANKING ELO NA KAZDEJ NAWIERZCHNI ---

ranking_naw = pd.Series(elo_naw)

for nazwa in ["Hard", "Clay", "Grass"]:
    # wybieramy klucze konczace sie dana nawierzchnia
    wybrane = ranking_naw[ranking_naw.index.str.endswith("_" + nazwa)]
    wybrane = wybrane.sort_values(ascending=False).head(10)
    print()
    print("=== TOP 10 ELO – " + nazwa + " ===")
    print(wybrane.round(0))

# --- WYKRES FORMY W CZASIE ---

hist = pd.DataFrame(historia)

# bierzemy 5 najlepszych zawodnikow z rankingu koncowego
najlepsi = ranking.head(5).index

plt.figure(figsize=(12, 6))
for gracz in najlepsi:
    dane_gracza = hist[hist["gracz"] == gracz]
    plt.plot(dane_gracza["data"], dane_gracza["elo"], label=gracz)

plt.title("Ranking Elo w czasie – 5 najlepszych zawodników")
plt.xlabel("Data")
plt.ylabel("Punkty Elo")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("wyniki/wykres_elo.png", dpi=120)
plt.close()

# --- ZAPIS ---

df.to_csv("wyniki/dane_z_elo.csv", index=False)

# zapisujemy koncowy ranking zeby mozna go bylo uzyc w porownaniach
ranking.to_csv("wyniki/ranking_elo.csv", header=["elo"])
ranking_naw.sort_values(ascending=False).to_csv("wyniki/ranking_elo_nawierzchnie.csv", header=["elo"])

print()
print("Zapisano wyniki/dane_z_elo.csv, ranking_elo.csv, ranking_elo_nawierzchnie.csv")
print("Zapisano wyniki/wykres_elo.png")
