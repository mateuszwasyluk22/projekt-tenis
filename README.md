# Analiza i prognoza meczów tenisa ATP

Projekt zaliczeniowy — **Systemy Sztucznej Inteligencji**

Autorzy: **Mateusz Wasyluk, Kacper Dobrowolski, Mikołaj Zdunek**

Rok akademicki 2026/2027

---

## O projekcie

Analiza meczów męskiego touru ATP w Pythonie i Pandas: czyszczenie danych,
analiza eksploracyjna, własny ranking formy zawodników (Elo) oraz trzy modele
uczenia maszynowego. Wyniki można obejrzeć w aplikacji webowej — wybiera się
dwóch zawodników i nawierzchnię, a program przewiduje przebieg hipotetycznego
meczu.

**Dane:** mecze ATP z lat 2021–2026 (Jeff Sackmann, CC BY-NC-SA 4.0).

**Technologie:** Python, Pandas, scikit-learn, Matplotlib, Streamlit.

---

## Modele uczenia maszynowego

Trzy modele, po jednym z każdej głównej rodziny.

| Plik | Rodzaj uczenia | Algorytm | Pytanie |
|---|---|---|---|
| `model.py` | nadzorowane — klasyfikacja | regresja logistyczna | Kto wygra mecz? |
| `model_asy.py` | nadzorowane — regresja | regresja liniowa | Ile asów zrobi serwujący? |
| `style.py` | nienadzorowane — klasteryzacja | KMeans + Gaussian Mixture | Jakie style gry istnieją? |

Pozostałe skrypty to przygotowanie danych i cech dla powyższych modeli.

Wspólne zasady: dane dzielimy na uczące i testowe **po czasie, nie losowo**,
żeby model nie znał przyszłości. Każdy model porównujemy z punktem odniesienia
— przy klasyfikacji jest nim rzut monetą, przy regresji zgadywanie średniej.

Szczegóły — cechy, wzory, sposób oceny — w dokumentacji.

---

## Uruchomienie

```bash
pip install -r requirements.txt
python narzedzia/pobierz.py
```

Kolejność ma znaczenie, każdy skrypt czyta plik stworzony przez poprzedni:

```bash
python projekt.py      # czyszczenie danych
python eda.py          # wykresy
python elo.py          # ranking Elo
python model.py        # klasyfikacja
python model_asy.py    # regresja
python style.py        # klasteryzacja
python top10.py        # TOP 10 wg nawierzchni
```

Aplikacja:

```bash
python -m streamlit run app.py
```

---

## Struktura

```
dane/          pliki CSV z meczami (nie trafiają do repozytorium)
wyniki/        wykresy, modele i pliki wynikowe
narzedzia/     skrypty pomocnicze do pobierania danych
*.py           skrypty analizy, w kolejności uruchamiania
app.py         aplikacja Streamlit
index.html     statyczny podgląd wykresów
```

---

## Dokumentacja

Pełny opis każdego skryptu, wzory, uzasadnienia decyzji, wyjaśnienie działania
uczenia maszynowego, słowniczek pojęć oraz wykaz użytych funkcji:
**[DOKUMENTACJA.md](DOKUMENTACJA.md)**

---

## Źródło danych

Jeff Sackmann, zbiór `tennis_atp`, przez mirror archiwalny
`Aneeshers/tennis-sackmann-archive`. Licencja **CC BY-NC-SA 4.0** — użycie
niekomercyjne, z podaniem autora.
