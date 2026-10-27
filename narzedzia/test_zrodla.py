import urllib.request

# rok testowy
rok = "2023"

# mozliwe adresy do mirrora danych
adresy = [
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_" + rok + ".csv",
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/master/atp/atp_matches_" + rok + ".csv",
    "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp_matches_" + rok + ".csv",
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main/atp/atp_matches_" + rok + ".csv",
]

for adres in adresy:
    try:
        odpowiedz = urllib.request.urlopen(adres, timeout=25)
        tresc = odpowiedz.read(300)
        print("DZIALA:", adres)
        print("poczatek pliku:", tresc[:200])
        print()
    except Exception as blad:
        print("BLAD  :", adres, "-", blad)
        print()
