import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("wyniki", exist_ok=True)

# wczytujemy mecze i rankingi elo
df = pd.read_csv("wyniki/dane_z_elo.csv")
ranking = pd.read_csv("wyniki/ranking_elo.csv", index_col=0)
ranking_naw = pd.read_csv("wyniki/ranking_elo_nawierzchnie.csv", index_col=0)


# --- POMOCNICZA: SZUKANIE ZAWODNIKA PO CZESCI NAZWISKA ---

def znajdz_zawodnika(tekst):
    # wszystkie nazwiska jakie sa w danych
    wszyscy = pd.concat([df["winner_name"], df["loser_name"]]).unique()
    wszyscy = pd.Series(wszyscy)

    # szukamy pasujacych, bez znaczenia male czy duze litery
    pasuje = wszyscy[wszyscy.str.lower().str.contains(tekst.lower())]

    if len(pasuje) == 0:
        print("Nie znaleziono zawodnika:", tekst)
        return None

    if len(pasuje) > 1:
        print("Pasuje kilku, wybieram pierwszego z:", list(pasuje)[:5])

    return pasuje.iloc[0]


# --- POMOCNICZA: STATYSTYKI SERWISU ZAWODNIKA ---

def statystyki(gracz):
    # mecze wygrane - bierzemy kolumny zwyciezcy
    jako_zw = df[df["winner_name"] == gracz]
    # mecze przegrane - bierzemy kolumny przegranego
    jako_prz = df[df["loser_name"] == gracz]

    asy = pd.concat([jako_zw["asy_zwyciezca"], jako_prz["asy_przegrany"]])
    bledy = pd.concat([jako_zw["bledy_zwyciezca"], jako_prz["bledy_przegrany"]])
    pierwszy = pd.concat([jako_zw["pierwszy_zwyciezca"], jako_prz["pierwszy_przegrany"]])

    return {
        "mecze": len(jako_zw) + len(jako_prz),
        "wygrane": len(jako_zw),
        "przegrane": len(jako_prz),
        "asy": asy.mean(),
        "bledy": bledy.mean(),
        "pierwszy_serwis": pierwszy.mean()
    }


# --- WPISANIE ZAWODNIKOW ---

print("Porównanie dwóch zawodników")
print("Wpisz fragment nazwiska, np. Djokovic")
print()

szukany_a = input("Zawodnik A: ")
szukany_b = input("Zawodnik B: ")

gracz_a = znajdz_zawodnika(szukany_a)
gracz_b = znajdz_zawodnika(szukany_b)

if gracz_a is None or gracz_b is None:
    exit()

print()
print("Porównuję:", gracz_a, "vs", gracz_b)

# --- BEZPOSREDNIE MECZE (HEAD TO HEAD) ---

# mecze gdzie grali ze soba, w obie strony
h2h_a = df[(df["winner_name"] == gracz_a) & (df["loser_name"] == gracz_b)]
h2h_b = df[(df["winner_name"] == gracz_b) & (df["loser_name"] == gracz_a)]

print()
print("=== BEZPOŚREDNIE MECZE ===")
print(gracz_a, "wygrał:", len(h2h_a))
print(gracz_b, "wygrał:", len(h2h_b))

# podzial na nawierzchnie
if len(h2h_a) + len(h2h_b) > 0:
    print()
    print("Według nawierzchni:")
    for nazwa in ["Hard", "Clay", "Grass"]:
        a = len(h2h_a[h2h_a["surface"] == nazwa])
        b = len(h2h_b[h2h_b["surface"] == nazwa])
        if a + b > 0:
            print("  ", nazwa, ":", a, "-", b)
else:
    print("(nie grali ze sobą w tym okresie)")

# --- STATYSTYKI OGOLNE ---

stat_a = statystyki(gracz_a)
stat_b = statystyki(gracz_b)

porownanie = pd.DataFrame({gracz_a: stat_a, gracz_b: stat_b})

print()
print("=== STATYSTYKI ===")
print(porownanie.round(2))

# --- RANKING ELO ---

elo_a = ranking.loc[gracz_a, "elo"]
elo_b = ranking.loc[gracz_b, "elo"]

print()
print("=== RANKING ELO ===")
print(gracz_a, ":", round(elo_a))
print(gracz_b, ":", round(elo_b))

# --- PROGNOZA NA KAZDEJ NAWIERZCHNI ---

print()
print("=== SZANSA NA WYGRANĄ WEDŁUG ELO ===")

# ogolna, bez wzgledu na nawierzchnie
szansa = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
print("Ogólnie:", gracz_a, round(szansa * 100, 1), "% -", round((1 - szansa) * 100, 1), "%", gracz_b)

# osobno dla kazdej nawierzchni
for nazwa in ["Hard", "Clay", "Grass"]:
    klucz_a = gracz_a + "_" + nazwa
    klucz_b = gracz_b + "_" + nazwa

    # jesli ktorys nie gral na tej nawierzchni to pomijamy
    if klucz_a not in ranking_naw.index or klucz_b not in ranking_naw.index:
        continue

    e_a = ranking_naw.loc[klucz_a, "elo"]
    e_b = ranking_naw.loc[klucz_b, "elo"]
    s = 1 / (1 + 10 ** ((e_b - e_a) / 400))

    print(nazwa, ":", round(s * 100, 1), "% -", round((1 - s) * 100, 1), "%")

# --- WYKRES PORoWNANIA ---

do_wykresu = porownanie.loc[["asy", "bledy", "pierwszy_serwis"]]

do_wykresu.plot(kind="bar", figsize=(9, 5))
plt.title("Porównanie: " + gracz_a + " vs " + gracz_b)
plt.ylabel("Wartość średnia")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("wyniki/wykres_porownanie.png", dpi=120)
plt.close()

print()
print("Zapisano wyniki/wykres_porownanie.png")
