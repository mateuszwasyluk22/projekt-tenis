import urllib.request
import os

# folder na dane
os.makedirs("dane", exist_ok=True)

# dziala mirror archiwalny, folder atp
BAZA = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/"

# lata ktore chcemy pobrac
lata = [2021, 2022, 2023, 2024, 2025, 2026]

for rok in lata:
    nazwa = "atp_matches_" + str(rok) + ".csv"
    url = BAZA + nazwa
    sciezka = "dane/" + nazwa

    try:
        urllib.request.urlretrieve(url, sciezka)
        rozmiar = os.path.getsize(sciezka)
        print("OK   ", nazwa, "-", round(rozmiar / 1024), "KB")
    except Exception as blad:
        print("BRAK ", nazwa, "-", blad)
        # usuwamy pusty plik jesli powstal
        if os.path.exists(sciezka):
            os.remove(sciezka)

print()
print("pliki w folderze dane:", os.listdir("dane"))
