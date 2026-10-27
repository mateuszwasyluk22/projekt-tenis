import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

# folder na wyniki
os.makedirs("wyniki", exist_ok=True)

# wczytujemy wszystkie pliki csv z folderu dane
pliki = glob.glob("dane/*.csv")
print("Znalezione pliki:", pliki)

tabele = []
for plik in pliki:
    tabele.append(pd.read_csv(plik))

# laczymy wszystko w jedna tabele
df = pd.concat(tabele)
print("Wszystkich meczów:", len(df))

# nawierzchnie ktore analizujemy
nawierzchnie = ["Hard", "Clay", "Grass"]

# minimalna liczba meczow zeby liczyc skutecznosc
minimum = 20

# przygotowujemy wykres - trzy panele obok siebie
fig, osie = plt.subplots(1, 3, figsize=(16, 6))

for i in range(len(nawierzchnie)):
    nazwa = nawierzchnie[i]

    # tylko mecze na tej nawierzchni
    mecze = df[df["surface"] == nazwa]

    # liczymy wygrane i przegrane kazdego zawodnika
    wygrane = mecze["winner_name"].value_counts()
    przegrane = mecze["loser_name"].value_counts()

    # suma rozegranych meczow
    razem = wygrane.add(przegrane, fill_value=0)

    # TOP 10 wedlug liczby wygranych
    top_wygrane = wygrane.head(10)

    print()
    print("=== " + nazwa + " – TOP 10 według liczby wygranych ===")
    print(top_wygrane)

    # skutecznosc w procentach, tylko dla grajacych czesto
    skutecznosc = wygrane / razem * 100
    skutecznosc = skutecznosc[razem >= minimum]
    top_skutecznosc = skutecznosc.sort_values(ascending=False).head(10)

    print()
    print("=== " + nazwa + " – TOP 10 według skuteczności (min " + str(minimum) + " meczów) ===")
    print(top_skutecznosc.round(1))

    # rysujemy slupki poziome
    top_wygrane.sort_values().plot(kind="barh", ax=osie[i], color="steelblue")
    osie[i].set_title(nazwa + " – liczba wygranych")
    osie[i].set_xlabel("Wygrane mecze")

# zapisujemy wykres do pliku
plt.tight_layout()
plt.savefig("wyniki/wykres_top10.png", dpi=120)
print()
print("Zapisano wyniki/wykres_top10.png")
