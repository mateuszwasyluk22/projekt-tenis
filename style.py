import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

os.makedirs("wyniki", exist_ok=True)

# --- ANALIZA STYLOW GRY ---
# Trzy spojrzenia na to samo:
# 1. KMeans - kazdy zawodnik trafia do JEDNEJ grupy
# 2. Role - zawodnik moze miec KILKA etykiet naraz (albo zadnej)
# 3. GaussianMixture - wersja "miekka", podaje procentowa przynaleznosc

df = pd.read_csv("wyniki/dane_czyste.csv")

MINIMUM = 30
ILE_GRUP = 4

# --- STATYSTYKI KAZDEGO ZAWODNIKA ---

a = pd.DataFrame({
    "gracz": df["winner_name"],
    "asy": df["asy_zwyciezca"],
    "bledy": df["bledy_zwyciezca"],
    "pierwszy_serwis": df["pierwszy_zwyciezca"],
    "asy_stracone": df["asy_przegrany"]
})

b = pd.DataFrame({
    "gracz": df["loser_name"],
    "asy": df["asy_przegrany"],
    "bledy": df["bledy_przegrany"],
    "pierwszy_serwis": df["pierwszy_przegrany"],
    "asy_stracone": df["asy_zwyciezca"]
})

dlugi = pd.concat([a, b])

tabela = dlugi.groupby("gracz").mean()
liczba_meczow = dlugi.groupby("gracz").size()

tabela = tabela[liczba_meczow >= MINIMUM]
tabela["mecze"] = liczba_meczow[liczba_meczow >= MINIMUM]

print("Zawodników do analizy:", len(tabela), "(minimum", MINIMUM, "meczów)")

cechy = ["asy", "bledy", "pierwszy_serwis", "asy_stracone"]

skaler = StandardScaler()
X = skaler.fit_transform(tabela[cechy])

# --- METODA LOKCIA ---

bledy_grup = []
zakres = range(2, 9)

for k in zakres:
    probny = KMeans(n_clusters=k, random_state=42, n_init=10)
    probny.fit(X)
    bledy_grup.append(probny.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(zakres), bledy_grup, marker="o")
plt.xlabel("Liczba grup")
plt.ylabel("Suma odległości wewnątrz grup")
plt.title("Metoda łokcia - wybór liczby grup")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("wyniki/wykres_lokiec.png", dpi=120)
plt.close()

# --- 1. KMEANS: JEDNA GRUPA NA ZAWODNIKA ---

model = KMeans(n_clusters=ILE_GRUP, random_state=42, n_init=10)
tabela["grupa"] = model.fit_predict(X)

# srodki grup w normalnej skali
srodki = pd.DataFrame(skaler.inverse_transform(model.cluster_centers_), columns=cechy)

print()
print("=== ŚRODKI GRUP ===")
print(srodki.round(2))

# --- NAZWY GRUP LICZONE Z DANYCH ---
# Nie przyklejamy sztywnej listy nazw. Patrzymy, czy grupa jest powyzej
# czy ponizej sredniej na dwoch osiach: serwisu i returnu.
# UWAGA: przy asach straconych MNIEJ znaczy LEPIEJ.

srednia_asow = srodki["asy"].mean()
srednia_strat = srodki["asy_stracone"].mean()

nazwy = {}

for i in range(len(srodki)):
    mocny_serwis = srodki.loc[i, "asy"] > srednia_asow
    mocny_return = srodki.loc[i, "asy_stracone"] < srednia_strat

    if mocny_serwis and mocny_return:
        nazwa = "Serwis i return"
    elif mocny_serwis:
        nazwa = "Serwisanci"
    elif mocny_return:
        nazwa = "Returnerzy"
    else:
        nazwa = "Uniwersalni"

    # jesli taka nazwa juz jest, dodajemy numer zeby sie nie dublowaly
    if nazwa in nazwy.values():
        nazwa = nazwa + " II"

    nazwy[i] = nazwa

tabela["styl"] = tabela["grupa"].map(nazwy)

print()
print("=== LICZEBNOŚĆ GRUP ===")
print(tabela["styl"].value_counts())

# --- 2. ROLE: ZAWODNIK MOZE MIEC KILKA NARAZ ---
# Progi liczymy z percentyli, czyli z rozkladu w naszym zbiorze.
# Gorne 25 procent w asach to "serwisant" i tak dalej.

prog_asy = tabela["asy"].quantile(0.75)
prog_return = tabela["asy_stracone"].quantile(0.25)
prog_stabilny = tabela["bledy"].quantile(0.25)
prog_ryzyko = tabela["bledy"].quantile(0.75)
prog_celny = tabela["pierwszy_serwis"].quantile(0.75)

print()
print("=== PROGI ROL ===")
print("Serwisant: asy powyżej", round(prog_asy, 2))
print("Mocny return: asy stracone poniżej", round(prog_return, 2))
print("Stabilny serwis: podwójne błędy poniżej", round(prog_stabilny, 2))
print("Ryzykowny serwis: podwójne błędy powyżej", round(prog_ryzyko, 2))
print("Celny pierwszy serwis: powyżej", round(prog_celny, 2), "%")

role_wszystkich = []

for gracz in tabela.index:
    wiersz = tabela.loc[gracz]
    role = []

    if wiersz["asy"] >= prog_asy:
        role.append("Serwisant")
    if wiersz["asy_stracone"] <= prog_return:
        role.append("Mocny return")
    if wiersz["bledy"] <= prog_stabilny:
        role.append("Stabilny serwis")
    if wiersz["bledy"] >= prog_ryzyko:
        role.append("Ryzykowny serwis")
    if wiersz["pierwszy_serwis"] >= prog_celny:
        role.append("Celny pierwszy serwis")

    # brak wyroznikow to tez informacja
    if len(role) == 0:
        role.append("Bez wyraźnych wyróżników")

    role_wszystkich.append(", ".join(role))

tabela["role"] = role_wszystkich

print()
print("=== NAJCZĘSTSZE ZESTAWY RÓL ===")
print(tabela["role"].value_counts().head(10))

# ilu zawodnikow ma wiecej niz jedna role
ile_rol = tabela["role"].str.count(",") + 1
print()
print("Zawodników z dwiema lub więcej rolami:", int((ile_rol >= 2).sum()))

# --- 3. GAUSSIAN MIXTURE: PRZYNALEZNOSC PROCENTOWA ---
# KMeans mowi "jestes w grupie 2". GMM mowi "jestes w 70% grupa 2, w 25% grupa 1".
# To odpowiedz na pytanie, czy ktos moze byc po trochu w dwoch stylach.

mieszanka = GaussianMixture(n_components=ILE_GRUP, random_state=42)
mieszanka.fit(X)
prawdopodobienstwa = mieszanka.predict_proba(X)

# nazwy dla skladnikow GMM, ta sama zasada co wyzej
srodki_gmm = pd.DataFrame(skaler.inverse_transform(mieszanka.means_), columns=cechy)

nazwy_gmm = {}
for i in range(len(srodki_gmm)):
    mocny_serwis = srodki_gmm.loc[i, "asy"] > srodki_gmm["asy"].mean()
    mocny_return = srodki_gmm.loc[i, "asy_stracone"] < srodki_gmm["asy_stracone"].mean()

    if mocny_serwis and mocny_return:
        nazwa = "Serwis i return"
    elif mocny_serwis:
        nazwa = "Serwisanci"
    elif mocny_return:
        nazwa = "Returnerzy"
    else:
        nazwa = "Uniwersalni"

    if nazwa in nazwy_gmm.values():
        nazwa = nazwa + " II"

    nazwy_gmm[i] = nazwa

glowny = []
glowny_proc = []
drugi = []
drugi_proc = []

for i in range(len(tabela)):
    szanse = pd.Series(prawdopodobienstwa[i])
    kolejnosc = szanse.sort_values(ascending=False)

    pierwszy_nr = kolejnosc.index[0]
    drugi_nr = kolejnosc.index[1]

    glowny.append(nazwy_gmm[pierwszy_nr])
    glowny_proc.append(round(kolejnosc.iloc[0] * 100, 1))
    drugi.append(nazwy_gmm[drugi_nr])
    drugi_proc.append(round(kolejnosc.iloc[1] * 100, 1))

tabela["styl_glowny"] = glowny
tabela["styl_glowny_proc"] = glowny_proc
tabela["styl_drugi"] = drugi
tabela["styl_drugi_proc"] = drugi_proc

# zawodnicy "na granicy" to ci, ktorych glowny styl nie przekracza 70 procent
na_granicy = tabela[tabela["styl_glowny_proc"] < 70]

print()
print("=== ZAWODNICY NA GRANICY DWÓCH STYLÓW ===")
print("Jest ich", len(na_granicy), "z", len(tabela))
print(na_granicy[["styl_glowny", "styl_glowny_proc", "styl_drugi", "styl_drugi_proc"]]
      .sort_values("styl_glowny_proc").head(10))

# --- KTO JEST W KTOREJ GRUPIE ---

for numer in sorted(nazwy.keys()):
    grupa = tabela[tabela["grupa"] == numer].sort_values("mecze", ascending=False)
    print()
    print("=== " + nazwy[numer] + " - najczęściej grający ===")
    print(grupa[["asy", "bledy", "pierwszy_serwis", "asy_stracone", "mecze"]].head(8).round(2))

# --- WYKRES ---

plt.figure(figsize=(10, 7))

for numer in sorted(nazwy.keys()):
    grupa = tabela[tabela["grupa"] == numer]
    plt.scatter(grupa["asy"], grupa["asy_stracone"], label=nazwy[numer], alpha=0.7, s=40)

plt.xlabel("Asy na 100 punktów serwisowych (serwis)")
plt.ylabel("Asy stracone na 100 punktów (return - mniej znaczy lepiej)")
plt.title("Style gry wykryte przez algorytm")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("wyniki/wykres_style.png", dpi=120)
plt.close()

# --- ZAPIS ---

tabela.to_csv("wyniki/style_gry.csv")

print()
print("Zapisano wyniki/style_gry.csv, wykres_style.png, wykres_lokiec.png")
