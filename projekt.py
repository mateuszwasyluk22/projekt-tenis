import pandas as pd
import glob
import os

# tworzymy folder na wyniki jesli go nie ma
os.makedirs("wyniki", exist_ok=True)

# --- KROK 1: WCZYTANIE DANYCH ---

# szukamy wszystkich plikow csv w folderze dane
pliki = glob.glob("dane/*.csv")
print("Znalezione pliki:", pliki)

# jesli folder dane jest pusty to nie ma co liczyc
if len(pliki) == 0:
    print()
    print("BŁĄD: folder dane/ jest pusty.")
    print("Najpierw pobierz dane: python narzedzia/pobierz.py")
    exit()

tabele = []
for plik in pliki:
    tabele.append(pd.read_csv(plik))

# laczymy wszystkie roczniki w jedna tabele
df = pd.concat(tabele)
print("Meczów na starcie:", len(df))

# --- KROK 2: SPRAWDZENIE KOLUMN ---

# kolumny ktorych bedziemy uzywac
potrzebne = ["tourney_date", "surface", "score", "winner_name", "loser_name",
             "winner_rank", "loser_rank", "winner_ht", "loser_ht",
             "winner_hand", "loser_hand",
             "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
             "w_1stIn", "l_1stIn"]

brakujace = []
for kolumna in potrzebne:
    if kolumna not in df.columns:
        brakujace.append(kolumna)

print("Brakujące kolumny:", brakujace)

# --- KROK 3: CZYSZCZENIE ---

# zamieniamy date turnieju na prawdziwa date
df["data"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

# sortujemy mecze od najstarszego do najnowszego
df = df.sort_values("data")

# usuwamy mecze przerwane: krecz, walkower, dyskwalifikacja
df["score"] = df["score"].astype(str)
df = df[~df["score"].str.contains("RET|W/O|DEF", na=False)]
print("Po usunięciu kreczów:", len(df))

# usuwamy mecze bez statystyk serwisu
statystyki = ["w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt", "w_1stIn", "l_1stIn"]
df = df.dropna(subset=statystyki)
print("Po usunięciu braków w statystykach:", len(df))

# usuwamy mecze z zerowa liczba punktow serwisowych (blad w danych)
df = df[df["w_svpt"] > 0]
df = df[df["l_svpt"] > 0]
print("Po usunięciu błędów:", len(df))

# --- KROK 4: NOWE KOLUMNY ---

# asy na 100 punktow serwisowych - uczciwe porownanie miedzy meczami
df["asy_zwyciezca"] = df["w_ace"] / df["w_svpt"] * 100
df["asy_przegrany"] = df["l_ace"] / df["l_svpt"] * 100

# podwojne bledy na 100 punktow serwisowych
df["bledy_zwyciezca"] = df["w_df"] / df["w_svpt"] * 100
df["bledy_przegrany"] = df["l_df"] / df["l_svpt"] * 100

# procent trafionych pierwszych serwisow
df["pierwszy_zwyciezca"] = df["w_1stIn"] / df["w_svpt"] * 100
df["pierwszy_przegrany"] = df["l_1stIn"] / df["l_svpt"] * 100

# --- KROK 5: PODSUMOWANIE I ZAPIS ---

print()
print("=== ZAKRES DAT ===")
print("Od", df["data"].min(), "do", df["data"].max())

print()
print("=== NAWIERZCHNIE ===")
print(df["surface"].value_counts(dropna=False))

print()
print("=== STATYSTYKI OPISOWE ===")
print(df[["asy_zwyciezca", "bledy_zwyciezca", "pierwszy_zwyciezca"]].describe().round(2))

# zapisujemy oczyszczone dane
df.to_csv("wyniki/dane_czyste.csv", index=False)
print()
print("Zapisano wyniki/dane_czyste.csv –", len(df), "meczów")
