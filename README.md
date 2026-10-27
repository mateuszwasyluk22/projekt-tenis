# Analiza i prognoza meczów tenisa ATP

Projekt zaliczeniowy — **Systemy Sztucznej Inteligencji**

Autorzy: **Mateusz Wasyluk, Kacper Dobrowolski, Mikołaj Zdunek**

Rok akademicki 2026/2027

---

## O czym jest projekt

Analizujemy mecze męskiego touru ATP: czyścimy dane, badamy zależności między
statystykami serwisu a wynikiem, budujemy własny ranking formy zawodników
i trenujemy trzy modele uczenia maszynowego. Całość można obejrzeć w aplikacji
webowej, w której wybiera się dwóch zawodników i nawierzchnię, a program
przewiduje przebieg hipotetycznego meczu.

**Zbiór danych:** mecze ATP z lat 2021–2026, kilkanaście tysięcy spotkań,
po czyszczeniu. Źródło: Jeff Sackmann, licencja CC BY-NC-SA 4.0.

**Technologie:** Python, Pandas, scikit-learn, Matplotlib, Streamlit.

---

## Jak działa uczenie maszynowe w tym projekcie

> **Uczenie maszynowe** to poddziedzina sztucznej inteligencji. W tym projekcie
> nie piszemy żadnych reguł ręcznie — nie mówimy programowi „jeśli zawodnik ma
> lepszy ranking, to wygra". Zamiast tego pokazujemy mu tysiące rozegranych
> meczów, a on sam znajduje zależności.

### W pigułce — jak model się uczy

**1. Cechy i odpowiedź.** Każdy mecz zamieniamy na zestaw liczb (*cechy*),
np. różnicę Elo, różnicę rankingu, rodzaj nawierzchni. Do tego dokładamy
*odpowiedź*, której szukamy — czy zawodnik A wygrał.

**2. Trenowanie.** Model na starcie zgaduje. Dla każdego meczu porównuje swoją
odpowiedź z prawdziwą i koryguje swoje **wagi** — liczby mówiące, jak mocno
każda cecha wpływa na wynik. Powtarza to tysiące razy, aż błąd przestaje maleć.
Tak wygląda „uczenie się": dopasowywanie wag, żeby pomyłek było jak najmniej.

**3. Testowanie.** Kluczowa zasada: model sprawdzamy na meczach, **których nigdy
nie widział**. Inaczej mierzylibyśmy tylko, jak dobrze zapamiętał dane uczące,
a nie czy czegokolwiek się nauczył.

**4. Punkt odniesienia.** Sama liczba typu „65% trafień" nic nie znaczy bez
porównania. Przy zgadywaniu zwycięzcy punktem odniesienia jest 50%, czyli rzut
monetą.

### Dwa rodzaje uczenia, oba są w projekcie

**Uczenie nadzorowane** — znamy prawidłowe odpowiedzi i model uczy się je
odtwarzać. Jak nauka z kluczem odpowiedzi. Dzieli się na:
- **klasyfikację**, gdy odpowiedź to kategoria (kto wygra),
- **regresję**, gdy odpowiedź to liczba (ile asów).

**Uczenie nienadzorowane** — nie mamy żadnych odpowiedzi. Model dostaje same
dane i szuka w nich struktury. U nas: sam dzieli zawodników na style gry,
chociaż nikt nie powiedział mu, czym jest „serwisant".

### Trzy modele w projekcie

Po jednym z każdej z powyższych rodzin.

| Plik | Rodzaj uczenia | Algorytm | Odpowiada na pytanie |
|---|---|---|---|
| `model.py` | nadzorowane — klasyfikacja | regresja logistyczna | Kto wygra mecz? |
| `model_asy.py` | nadzorowane — regresja | regresja liniowa | Ile asów zrobi serwujący? |
| `style.py` | **nienadzorowane** — klasteryzacja | KMeans + Gaussian Mixture | Jakie style gry istnieją w stawce? |

Pozostałe skrypty (czyszczenie, wykresy, ranking Elo) to klasyczna analiza
danych i przygotowanie cech dla powyższych modeli.

### Model 1 — kto wygra mecz (klasyfikacja)

**Jak działa regresja logistyczna.** Każdej cesze przypisuje wagę, mnoży cechy
przez wagi i sumuje. Wynik przepuszcza przez funkcję, która ściska dowolną
liczbę do przedziału od 0 do 1 — i tak powstaje prawdopodobieństwo. Im większa
waga, tym mocniej dana cecha przesuwa wynik.

**Cechy:** różnica Elo, różnica Elo na danej nawierzchni, różnica rankingu ATP,
różnica wzrostu, rodzaj nawierzchni (kodowanie one-hot).

**Uczenie i test dzielone po czasie, nie losowo** — model uczy się na 80%
najstarszych meczów, a sprawdzany jest na 20% najnowszych. Przy podziale
losowym uczyłby się na meczach z grudnia i był testowany na styczniowych,
czyli znałby przyszłość.

**Ocena:** dokładność i macierz pomyłek. Punkt odniesienia to 50%, czyli rzut
monetą. Skrypt porównuje **trzy warianty** — sam ranking ATP, samo Elo oraz
wszystkie cechy razem — żeby pokazać, która informacja faktycznie coś wnosi.

### Model 2 — ile asów zrobi serwujący (regresja)

**Jak działa regresja liniowa.** Szuka prostej zależności: wynik to suma cech
pomnożonych przez wagi. Odpowiada na pytanie „o ile wzrośnie liczba asów, jeśli
serwujący jest o 10 cm wyższy". Różnica wobec modelu 1 jest taka, że tu wynik
jest liczbą, a nie prawdopodobieństwem.

**Cechy:** historia serwisu zawodnika, podatność przeciwnika na asy, wzrost
obu zawodników, ręka przeciwnika, jego ranking, nawierzchnia.

Z każdego meczu powstają **dwa wiersze** — raz serwuje zwycięzca, raz przegrany
— dzięki czemu model uczy się serwisu, a nie tego, kto wygrał.

**Średnie historyczne liczymy wyłącznie z okresu uczenia.** Gdyby policzyć je
z całości danych, średnia zawodnika zawierałaby mecze ze zbioru testowego
i model znałby przyszłość.

**Ocena:** średni błąd bezwzględny (MAE) i R². Skrypt wypisuje też błąd, jaki
dałoby zwykłe zgadywanie średniej — jeśli model tego nie bije, niczego się
nie nauczył.

### Model 3 — style gry (uczenie nienadzorowane)

**Jak działa KMeans.** Losuje kilka punktów jako środki grup, przypisuje każdego
zawodnika do najbliższego środka, przesuwa środki na średnią swojej grupy
i powtarza, aż nic się nie zmienia. Efekt: zawodnicy w jednej grupie są
do siebie podobni, a grupy są od siebie odległe.

Tu algorytm nie dostaje **żadnych etykiet**. Sam dzieli zawodników na grupy
na podstawie czterech cech: asów, podwójnych błędów, procentu pierwszego
serwisu i asów traconych przy returnie.

Dane są wcześniej **skalowane**, bo asy mają wartości rzędu 12, a procent
pierwszego serwisu rzędu 60 — bez tego algorytm patrzyłby głównie na większe
liczby.

Liczbę grup dobieramy **metodą łokcia**: rysujemy, jak maleje suma odległości
wewnątrz grup przy rosnącej liczbie grup, i szukamy miejsca, w którym krzywa
przestaje gwałtownie opadać.

Nazwy grup (Serwisanci, Returnerzy, Serwis i return, Uniwersalni) **nadajemy
my** — algorytm zwraca wyłącznie numery. Przypisujemy je automatycznie, badając
położenie środka każdej grupy względem średniej na osi serwisu i returnu.

Ponieważ twarde przypisanie do jednej grupy gubi informację, projekt pokazuje
jeszcze dwa ujęcia:

- **Role** — zawodnik może mieć kilka naraz (np. „Serwisant, Ryzykowny serwis").
  Progi wyznaczane percentylami: górne 25% w asach to serwisant i tak dalej.
- **Gaussian Mixture** — miękka wersja klasteryzacji, podaje przynależność
  procentową. Zamiast „jesteś w grupie 2" mówi „jesteś w 65% serwisantem,
  w 30% uniwersalnym".

---

## Jak liczone są kluczowe wielkości

### Normalizacja statystyk serwisu

```
asy na 100 punktów = liczba asów ÷ liczba punktów serwisowych × 100
```

Mecz pięciosetowy daje dwa razy więcej okazji na asa niż trzysetowy. Bez
przeliczenia model uczyłby się przede wszystkim, że dłuższe mecze mają więcej
asów. Po normalizacji mecze różnej długości stają się porównywalne.

*Przykład z danych:* 14 asów przy 81 punktach serwisowych to wskaźnik 17,3;
11 asów przy 63 punktach to 17,5. Surowo różnica wygląda na dużą — serwis był
niemal identyczny.

### Ranking Elo

Ranking ATP sumuje punkty za turnieje z ostatnich 12 miesięcy i nie patrzy,
z kim się grało. Elo aktualizuje się po każdym meczu i uwzględnia siłę
przeciwnika, więc lepiej oddaje bieżącą formę.

```
szansa_A = 1 / (1 + 10 ^ ((Elo_B − Elo_A) / 400))
zmiana   = K × (1 − szansa_zwycięzcy)          gdzie K = 32
```

Każdy zaczyna od 1500 punktów. Niespodzianki ważą więcej: faworyt z 90%
szans zyskuje za wygraną 3,2 punktu, outsider z 10% szans — 28,8.

Prowadzimy **dwa rankingi równolegle**: ogólny i osobny dla każdej
nawierzchni, co pozwala wychwycić specjalistów od mączki czy trawy.

Elo zapisywane do tabeli to zawsze stan **sprzed** meczu — inaczej model
dostałby informację o wyniku, którego ma dopiero szukać.

### Prognoza liczby asów

Model regresji zwraca wskaźnik na 100 punktów serwisowych, który następnie
mnożymy przez typową liczbę punktów serwisowych danego zawodnika, żeby otrzymać
przewidywaną liczbę asów w całym meczu.

---

## Przebieg analizy

```
dane/*.csv
    │
    ├─ projekt.py ──────► dane_czyste.csv          czyszczenie + normalizacja
    │                           │
    │                           ├─ eda.py ────────► 5 wykresów
    │                           │
    │                           ├─ elo.py ────────► dane_z_elo.csv + rankingi
    │                           │        │
    │                           │        ├─ model.py ──────► model_wygrana.pkl
    │                           │        │
    │                           ├─ model_asy.py ──────────► model_asy.pkl
    │                           │
    │                           └─ style.py ──────────────► style_gry.csv
    │
    └─ top10.py ───────► wykres_top10.png
                                │
                                └──────────► app.py  (aplikacja Streamlit)
```

Każdy etap czyta plik stworzony przez poprzedni, dlatego kolejność ma znaczenie.

---

## Uruchomienie

```bash
pip install -r requirements.txt
python narzedzia/pobierz.py      # pobranie danych
```

```bash
python projekt.py
python eda.py
python elo.py
python model.py
python model_asy.py
python style.py
python top10.py
```

Aplikacja:

```bash
python -m streamlit run app.py
```

Otwiera się w przeglądarce pod `localhost:8501`.

---

## Co pokazuje aplikacja

Po wybraniu dwóch zawodników i nawierzchni:

- **prognoza** — szansa na wygraną z modelu, zestawiona z wynikiem samego
  wzoru Elo, żeby było widać, co wnoszą dodatkowe cechy,
- **profil zawodnika** — uśrednione statystyki ogółem i osobno dla każdej
  nawierzchni,
- **ostatnie mecze** — forma z dziesięciu ostatnich spotkań,
- **bezpośrednie mecze** — bilans, statystyki z tych konkretnych meczów oraz
  porównanie „jak gra przeciwko temu rywalowi kontra jak gra ogólnie",
- **statystyki** — wykresy, w tym zależność między procentem pierwszego serwisu
  a liczbą asów wraz ze współczynnikiem korelacji,
- **style gry** — role, zawodnicy na granicy dwóch stylów, podział KMeans,
- **rankingi** — TOP 20 Elo, ogólne i na wybranej nawierzchni.

Jeśli zawodnik nie rozegrał meczu na wybranej nawierzchni, aplikacja pokazuje
**„brak danych"** zamiast wartości startowej 1500 i ostrzega, że prognoza jest
w tym wypadku niepewna.

---

## Ograniczenia

Świadomie odnotowane, żeby nie przypisywać modelowi możliwości, których nie ma.

**Czego nie ma w danych:** kierunku i rodzaju serwisu, ustawienia returnera,
podziału na halę i kort otwarty, warunków pogodowych, informacji o kontuzjach.
Dane o pozycjach zawodników pochodzą z systemów typu Hawk-Eye i nie są
publicznie dostępne.

**Bilans bezpośrednich meczów nie wchodzi do modelu.** Aplikacja go wyświetla,
ale większość par zagrała ze sobą 0–3 razy, więc cecha byłaby w większości
przypadków zerem.

**Styl gry to interpretacja, nie pomiar.** Model wykrywa podobieństwo
statystyk, nie sposób poruszania się po korcie.

**Prognoza nie uwzględnia** formy dziennej, zmęczenia po poprzednim meczu
ani kontuzji.

---

## Dokumentacja szczegółowa

Pełny opis każdego skryptu, wzory, uzasadnienia decyzji, słowniczek pojęć oraz
wykaz użytych funkcji Pandas i klas scikit-learn:
**[DOKUMENTACJA.md](DOKUMENTACJA.md)**

---

## Źródło danych i licencja

Dane: **Jeff Sackmann**, zbiór `tennis_atp`, pobrane przez mirror archiwalny
`Aneeshers/tennis-sackmann-archive`.

Licencja danych: **CC BY-NC-SA 4.0** — użycie niekomercyjne, z podaniem autora,
na tej samej licencji.
