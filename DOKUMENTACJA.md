# Analiza meczów tenisa ATP — dokumentacja projektu

Projekt zaliczeniowy z przedmiotu **Systemy Sztucznej Inteligencji**.

Analiza meczów ATP w Pythonie i Pandas: czyszczenie danych, analiza eksploracyjna,
własny ranking Elo, trzy modele uczenia maszynowego i aplikacja webowa.

---

## 1. Struktura projektu

```
projekt-tenis/
├── dane/                    pliki CSV z meczami (nie trafiają do repozytorium)
├── wyniki/                  wszystko, co generują skrypty
├── narzedzia/
│   ├── pobierz.py           pobiera roczniki z mirrora
│   └── test_zrodla.py       sprawdza, który adres mirrora działa
│
├── projekt.py               KROK 1 — wczytanie i czyszczenie
├── eda.py                   KROK 2 — analiza eksploracyjna i wykresy
├── elo.py                   KROK 3 — ranking Elo
├── model.py                 KROK 4 — klasyfikacja (kto wygra)
├── model_asy.py             KROK 5 — regresja (ile asów)
├── style.py                 KROK 6 — klasteryzacja (style gry)
├── top10.py                 dodatek — TOP 10 wg nawierzchni
├── porownaj.py              dodatek — porównanie dwóch zawodników w terminalu
├── app.py                   aplikacja Streamlit
├── index.html               statyczny podgląd wykresów
│
├── README.md                szybki start
├── DOKUMENTACJA.md          ten plik
├── requirements.txt
└── .gitignore
```

### Pliki tworzone automatycznie w `wyniki/`

| Plik | Tworzy go | Zawiera |
|---|---|---|
| `dane_czyste.csv` | projekt.py | oczyszczone mecze z przeliczonymi wskaźnikami |
| `dane_z_elo.csv` | elo.py | to samo + cztery kolumny Elo sprzed meczu |
| `ranking_elo.csv` | elo.py | końcowe Elo każdego zawodnika |
| `ranking_elo_nawierzchnie.csv` | elo.py | Elo osobno dla każdej nawierzchni |
| `model_wygrana.pkl` | model.py | wytrenowany model klasyfikacji |
| `model_asy.pkl` | model_asy.py | wytrenowany model regresji + średnie zawodników |
| `style_gry.csv` | style.py | grupa, role i przynależność procentowa |
| `wyniki_modelu.txt` | model.py | dokładność trzech wariantów |
| `wykres_*.png` | różne | dziewięć wykresów |

---

## 2. Uruchomienie

### Instalacja

```bash
pip install -r requirements.txt
```

Wymagane: `pandas`, `matplotlib`, `scikit-learn`. Streamlit tylko do aplikacji.

### Pobranie danych

```bash
python narzedzia/pobierz.py
```

### Kolejność — ma znaczenie

Każdy skrypt czyta plik stworzony przez poprzedni.

```bash
python projekt.py      # -> dane_czyste.csv
python eda.py          # -> 5 wykresów
python elo.py          # -> dane_z_elo.csv, rankingi
python model.py        # -> model_wygrana.pkl
python model_asy.py    # -> model_asy.pkl
python style.py        # -> style_gry.csv
python top10.py        # niezależny, czyta prosto z dane/
```

### Aplikacja

```bash
python -m streamlit run app.py
```

Jeśli komenda `streamlit` nie jest rozpoznawana, to kwestia zmiennej PATH —
`python -m streamlit` omija ten problem.

---

## 3. Dane źródłowe

### Pochodzenie

Dane: **Jeff Sackmann**, zbiór `tennis_atp`.

Oryginalne repozytorium `JeffSackmann/tennis_atp` przestało działać (zwraca 404).
Korzystamy z mirrora archiwalnego `Aneeshers/tennis-sackmann-archive`, folder `atp`.
Licencja: **CC BY-NC-SA 4.0** — użycie niekomercyjne, z podaniem autora.

### Kolumny, których używamy

| Kolumna | Znaczenie |
|---|---|
| `tourney_name` | nazwa turnieju |
| `tourney_date` | data turnieju, format RRRRMMDD |
| `surface` | nawierzchnia: Hard, Clay, Grass |
| `score` | wynik setowy, zawiera `RET` przy kreczu |
| `winner_name` / `loser_name` | nazwiska |
| `winner_rank` / `loser_rank` | pozycja w rankingu ATP przed meczem |
| `winner_ht` / `loser_ht` | wzrost w centymetrach |
| `winner_hand` / `loser_hand` | `R` prawa ręka, `L` lewa |
| `w_ace` / `l_ace` | asy serwisowe |
| `w_df` / `l_df` | podwójne błędy |
| `w_svpt` / `l_svpt` | punkty rozegrane przy własnym serwisie |
| `w_1stIn` / `l_1stIn` | trafione pierwsze serwisy |

Prefiks `w_` oznacza zwycięzcę, `l_` przegranego.

> **Uwaga strukturalna.** W danych nie ma „zawodnika A i B", tylko „zwycięzcę
> i przegranego". To wraca w kilku miejscach projektu — za każdym razem, gdy
> chcemy policzyć coś dla konkretnego zawodnika, musimy sprawdzić, po której
> stronie stał w danym meczu.

---

## 4. KROK 1 — `projekt.py` (czyszczenie)

### Co robi po kolei

1. **Wczytuje wszystkie CSV** z folderu `dane/` przez `glob.glob()` i skleja
   funkcją `pd.concat()`. Jeśli folder jest pusty, kończy z czytelnym komunikatem.
2. **Sprawdza kolumny** — wypisuje listę brakujących. Pusta lista oznacza, że
   format danych się zgadza.
3. **Zamienia datę** z liczby `20230115` na prawdziwy typ daty.
4. **Sortuje chronologicznie.** Kluczowe dla Elo — ranking musi być liczony
   w kolejności rozgrywania meczów.
5. **Usuwa mecze przerwane** — zawierające `RET` (krecz), `W/O` (walkower),
   `DEF` (dyskwalifikacja). Mają niepełne statystyki serwisu.
6. **Usuwa mecze bez statystyk serwisu** przez `dropna()`.
7. **Usuwa rekordy z zerową liczbą punktów serwisowych** — inaczej dzielenie
   przez zero.
8. **Dolicza wskaźniki** (niżej).
9. **Zapisuje** `wyniki/dane_czyste.csv`.

### Normalizacja — najważniejsza decyzja w projekcie

```python
df["asy_zwyciezca"] = df["w_ace"] / df["w_svpt"] * 100
```

Wzór: **liczba asów ÷ liczba punktów serwisowych × 100**.

Czyli: ile asów przypada na każde 100 zaserwowanych punktów.

**Dlaczego.** Mecz pięciosetowy daje dwa razy więcej okazji na asa niż
trzysetowy. Gdyby model dostał surową liczbę asów, nauczyłby się przede
wszystkim tego, że *dłuższe mecze mają więcej asów* — co jest prawdą
bezużyteczną. Przeliczenie na 100 punktów sprawia, że mecze różnej długości
stają się porównywalne.

**Przykład z danych.** Zawodnik zrobił 14 asów przy 81 punktach serwisowych
(wskaźnik 17,3) oraz 11 asów przy 63 punktach (wskaźnik 17,5). Surowo wygląda
to na sporą różnicę, a serwował praktycznie tak samo — drugi mecz był po prostu
krótszy.

Fachowa nazwa tego zabiegu to **normalizacja**.

### Wszystkie dodane kolumny

| Kolumna | Wzór | Znaczenie |
|---|---|---|
| `asy_zwyciezca` | `w_ace / w_svpt * 100` | asy na 100 pkt serwisowych |
| `asy_przegrany` | `l_ace / l_svpt * 100` | to samo dla przegranego |
| `bledy_zwyciezca` | `w_df / w_svpt * 100` | podwójne błędy na 100 pkt |
| `bledy_przegrany` | `l_df / l_svpt * 100` | to samo dla przegranego |
| `pierwszy_zwyciezca` | `w_1stIn / w_svpt * 100` | % trafionych pierwszych serwisów |
| `pierwszy_przegrany` | `l_1stIn / l_svpt * 100` | to samo dla przegranego |

Ostatni wskaźnik czyta się inaczej niż pozostałe — to po prostu **procent
pierwszych podań, które weszły w pole serwisowe**. Typowy zakres na tourze
to mniej więcej 55–68%.

---

## 5. KROK 2 — `eda.py` (analiza eksploracyjna)

Czyta `dane_czyste.csv`, tworzy pięć wykresów i wypisuje odpowiadające im tabele.

| Plik wyjściowy | Co pokazuje | Funkcje Pandas |
|---|---|---|
| `wykres_nawierzchnie.png` | liczba meczów na każdej nawierzchni | `value_counts()` |
| `wykres_asy.png` | średnie asy wg nawierzchni, zwycięzcy i przegrani osobno | `groupby().mean()` |
| `wykres_ranking.png` | TOP 10 zawodników wg liczby wygranych | `value_counts().head()` |
| `wykres_korelacje.png` | macierz korelacji sześciu wskaźników | `corr()` |
| `wykres_reka.png` | asy serwującego wg ręki przeciwnika i nawierzchni | `groupby().unstack()` |

### Jak czytać macierz korelacji

Wartości od −1 do 1:
- blisko **1** — silny związek dodatni (jedno rośnie, drugie też),
- blisko **−1** — silny ujemny (jedno rośnie, drugie maleje),
- blisko **0** — brak zależności liniowej.

Na przekątnej zawsze 1, bo każda kolumna koreluje sama ze sobą idealnie.

**Korelacja nie dowodzi przyczyny.** Dwie wartości mogą rosnąć razem, bo obie
zależą od czegoś trzeciego — na przykład od tego, że zawodnik miał dobry dzień
serwisowy.

### Hipoteza o leworęcznych

Wykres `wykres_reka.png` sprawdza konkretne pytanie: *czy serwujący zdobywa
więcej asów przeciwko leworęcznym?* Wynik może wyjść w dowolną stronę — to
hipoteza do zweryfikowania, nie teza do udowodnienia. Jeśli różnicy nie ma,
to też jest wynik i należy go uczciwie opisać.

---

## 6. KROK 3 — `elo.py` (ranking Elo)

### Czym Elo różni się od rankingu ATP

**Ranking ATP** sumuje punkty za turnieje z ostatnich 12 miesięcy. Zawodnik,
który wygrał duży turniej rok temu i od tego czasu przegrywa, wciąż ma dużo
punktów — ranking reaguje z opóźnieniem i nie patrzy, z kim się grało.

**Elo** aktualizuje się po **każdym meczu** i uwzględnia siłę przeciwnika.
Dlatego lepiej oddaje bieżącą formę. System pochodzi z szachów.

### Wzory

**1. Oczekiwana szansa zawodnika A przed meczem:**

```
szansa_A = 1 / (1 + 10 ^ ((Elo_B − Elo_A) / 400))
```

Przy równym Elo wychodzi 0,5, czyli 50%. Przewaga 400 punktów daje mniej
więcej 10:1 na korzyść lepszego.

**2. Zmiana po meczu:**

```
zmiana = K × (1 − szansa_zwycięzcy)
```

Zwycięzca zyskuje dokładnie tyle, ile traci przegrany — suma punktów
w systemie się nie zmienia.

### Parametry

- **Start: 1500 punktów** dla każdego nowego zawodnika.
- **K = 32** — mówi, jak mocno pojedynczy mecz rusza rankingiem. Większe K
  oznacza szybszą reakcję na formę, ale też większe skoki po przypadkowych
  wynikach.

### Niespodzianki ważą więcej

To sedno mechanizmu. Jeśli faworyt miał 90% szans i wygrał, zyskuje
`32 × (1 − 0,9) = 3,2` punktu. Jeśli wygrał outsider z szansą 10%, zyskuje
`32 × (1 − 0,1) = 28,8` punktu. Algorytm „dziwi się" i mocniej koryguje ocenę.

### Elo osobno dla każdej nawierzchni

Skrypt prowadzi **dwa rankingi równolegle**: ogólny i osobny dla każdej
nawierzchni. Klucz w słowniku to `"Nazwisko_Hard"`. Dzięki temu widać
specjalistów od mączki czy trawy.

### Co zapisuje

- `dane_z_elo.csv` — dane z czterema kolumnami Elo sprzed meczu
- `ranking_elo.csv` — końcowe Elo każdego zawodnika
- `ranking_elo_nawierzchnie.csv` — Elo w rozbiciu na nawierzchnie
- `wykres_elo.png` — forma pięciu najlepszych w czasie

---

## 6a. Jak działa uczenie maszynowe w tym projekcie

**Uczenie maszynowe** to poddziedzina sztucznej inteligencji. Nie piszemy
żadnych reguł ręcznie — nie mówimy programowi „jeśli zawodnik ma lepszy
ranking, to wygra". Pokazujemy mu tysiące rozegranych meczów, a on sam
znajduje zależności.

### Jak model się uczy — cztery kroki

**1. Cechy i odpowiedź.** Każdy mecz zamieniamy na zestaw liczb (*cechy*),
np. różnicę Elo, różnicę rankingu, rodzaj nawierzchni. Do tego dokładamy
*odpowiedź*, której szukamy — czy zawodnik A wygrał.

**2. Trenowanie.** Model na starcie zgaduje. Dla każdego meczu porównuje swoją
odpowiedź z prawdziwą i koryguje **wagi** — liczby mówiące, jak mocno każda
cecha wpływa na wynik. Powtarza to tysiące razy, aż błąd przestaje maleć.
Tak wygląda „uczenie się": dopasowywanie wag, żeby pomyłek było jak najmniej.

**3. Testowanie.** Model sprawdzamy na meczach, **których nigdy nie widział**.
Inaczej mierzylibyśmy tylko, jak dobrze zapamiętał dane uczące.

**4. Punkt odniesienia.** Liczba typu „65% trafień" nic nie znaczy bez
porównania. Przy zgadywaniu zwycięzcy punktem odniesienia jest 50%, czyli
rzut monetą.

### Dwa rodzaje uczenia, oba w projekcie

**Uczenie nadzorowane** — znamy prawidłowe odpowiedzi i model uczy się je
odtwarzać. Dzieli się na **klasyfikację** (odpowiedź to kategoria: kto wygra)
i **regresję** (odpowiedź to liczba: ile asów).

**Uczenie nienadzorowane** — nie mamy żadnych odpowiedzi. Model dostaje same
dane i szuka w nich struktury. U nas: sam dzieli zawodników na style gry,
chociaż nikt nie powiedział mu, czym jest „serwisant".

### Algorytmy w jednym zdaniu

**Regresja logistyczna** — każdej cesze przypisuje wagę, mnoży cechy przez wagi
i sumuje; wynik przepuszcza przez funkcję ściskającą dowolną liczbę do
przedziału od 0 do 1, i tak powstaje prawdopodobieństwo.

**Regresja liniowa** — to samo mnożenie i sumowanie, ale wynikiem jest zwykła
liczba. Odpowiada na pytanie „o ile wzrośnie liczba asów, jeśli serwujący jest
o 10 cm wyższy".

**KMeans** — losuje kilka punktów jako środki grup, przypisuje każdego
zawodnika do najbliższego środka, przesuwa środki na średnią swojej grupy
i powtarza, aż nic się nie zmienia.

**Gaussian Mixture** — miękka wersja KMeans. Zamiast przypisać do jednej grupy,
podaje procentową przynależność do każdej.

---

## 7. KROK 4 — `model.py` (klasyfikacja: kto wygra)

Typ problemu: **klasyfikacja binarna**. Odpowiedź to „tak" albo „nie".
Algorytm: **regresja logistyczna**.

### Problem: zwycięzca jest zawsze pierwszy

Kolumna `winner_name` to z definicji zwycięzca. Gdyby karmić model tak
ułożonymi danymi, odpowiedź siedziałaby w samym ustawieniu kolumn.

**Rozwiązanie:** co drugi mecz odwracamy.

```python
if i % 2 == 0:
    wynik = 1    # zawodnik A to zwycięzca
else:
    wynik = 0    # zawodnik A to przegrany
```

Dzięki temu obie klasy mają mniej więcej po połowie przypadków, a model musi
faktycznie czegoś się nauczyć.

### Cechy podawane modelowi

| Cecha | Jak liczona |
|---|---|
| `roznica_elo` | Elo zawodnika A minus Elo zawodnika B |
| `roznica_elo_naw` | to samo, ale Elo na danej nawierzchni |
| `roznica_rank` | ranking B minus ranking A (niższy numer = lepszy) |
| `roznica_wzrost` | wzrost A minus wzrost B, w centymetrach |
| `twarda`, `ziemia`, `trawa` | nawierzchnia jako trzy kolumny 0/1 |

Nawierzchnia jest tekstem, a model przyjmuje tylko liczby. Zamiana na kolumny
zero-jedynkowe nazywa się **kodowaniem one-hot**.

### Podział na uczenie i test — po czasie, nie losowo

```python
granica = int(len(dane) * 0.8)
uczenie = dane.iloc[:granica]    # 80% najstarszych meczów
test = dane.iloc[granica:]       # 20% najnowszych
```

**Dlaczego nie losowo.** Przy losowym podziale model uczyłby się na meczach
z grudnia i był sprawdzany na meczach ze stycznia — czyli znałby przyszłość.
Podział po czasie odwzorowuje realną sytuację: uczymy się na przeszłości,
przewidujemy przyszłość.

### Trzy warianty do porównania

1. **Tylko ranking ATP** — punkt odniesienia
2. **Tylko Elo** — czy Elo jest lepsze od rankingu ATP?
3. **Wszystko razem** — czy dodatkowe cechy pomagają?

To porównanie jest mocniejsze niż podanie jednej liczby, bo odpowiada na
konkretne pytanie badawcze.

### Jak czytać wyniki

- **Dokładność** — procent poprawnie przewidzianych meczów. Punkt odniesienia
  to 50%, czyli rzut monetą. Sensowne modele w tenisie osiągają zwykle 60–70%.
- **Macierz pomyłek** — tabela 2×2 z poprawnymi i błędnymi przewidywaniami dla
  obu klas. Pokazuje, czy model nie zgaduje ciągle tak samo.
- **Wagi modelu** — im większa wartość bezwzględna, tym mocniej cecha wpływa
  na wynik. Znak mówi o kierunku.
- **`predict_proba`** — zamiast „wygra A" daje prawdopodobieństwo, np. 64%.

### Co zapisuje

`model_wygrana.pkl` — wytrenowany model plus lista cech, w formacie `pickle`.
Aplikacja go wczytuje, zamiast trenować od nowa przy każdym uruchomieniu.

---

## 8. KROK 5 — `model_asy.py` (regresja: ile asów)

Typ problemu: **regresja**. Odpowiedź to liczba, nie „tak/nie".
Algorytm: **regresja liniowa**.

### Dwa wiersze z jednego meczu

Każdy mecz daje dwa przypadki: raz serwuje zwycięzca, raz przegrany. Dzięki
temu model uczy się serwisu, a nie tego, kto wygrał mecz.

### Cechy

| Cecha | Jak liczona |
|---|---|
| `serwis_historia` | średnia asów serwującego z okresu uczenia |
| `straty_przeciwnika` | średnia asów traconych przez przeciwnika |
| `wzrost_serwujacego` | w centymetrach |
| `wzrost_przeciwnika` | w centymetrach |
| `przeciwnik_leworeczny` | 1 jeśli lewa ręka, 0 jeśli prawa |
| `ranking_przeciwnika` | pozycja ATP |
| `twarda`, `ziemia`, `trawa` | kodowanie one-hot |

### Średnie historyczne liczone tylko z okresu uczenia

Dwie pierwsze cechy to średnie historyczne — i tu czai się pułapka.

```python
srednia_ogolna = uczenie["asy"].mean()
serwis_gracza = uczenie.groupby("serwujacy")["asy"].mean()
```

Średnie liczymy **wyłącznie z okresu uczenia**, nigdy z całości. Gdybyśmy użyli
wszystkich danych, średnia zawodnika zawierałaby mecze ze zbioru testowego —
czyli model znałby przyszłość. Zawodnik, który nie wystąpił w okresie uczenia,
dostaje średnią ogólną.

### Jak oceniamy wynik

- **MAE** (średni błąd bezwzględny) — o ile średnio mylimy się w asach na
  100 punktów. Im mniej, tym lepiej.
- **R²** — ile zmienności danych model wyjaśnia. 1 to ideał, 0 oznacza tyle,
  co zgadywanie średniej, wartość ujemna — gorzej niż zgadywanie średniej.
- **Punkt odniesienia** — skrypt wypisuje też błąd, jaki dałoby samo zgadywanie
  średniej. Jeśli model tego nie bije, niczego się nie nauczył i trzeba to
  uczciwie napisać.

Wykres `wykres_model_asy.png` zestawia przewidywania z rzeczywistością.
Czerwona linia to idealne trafienie — im bliżej niej punkty, tym lepiej.

---

## 9. KROK 6 — `style.py` (klasteryzacja: style gry)

Typ problemu: **uczenie nienadzorowane**. Nie podajemy algorytmowi żadnych
etykiet — on sam szuka podobieństw.

Skrypt daje **trzy spojrzenia na to samo pytanie**.

### Przygotowanie

- Bierzemy tylko zawodników z **minimum 30 meczami** (`MINIMUM`), żeby średnie
  były wiarygodne.
- Cztery cechy: `asy`, `bledy`, `pierwszy_serwis`, `asy_stracone`.
- **Skalowanie** przez `StandardScaler`. Konieczne, bo asy mają wartości rzędu
  12, a procent pierwszego serwisu rzędu 60 — bez skalowania algorytm patrzyłby
  głównie na tę drugą wielkość, tylko dlatego że liczby są większe.

### Spojrzenie 1 — KMeans: jedna grupa na zawodnika

Algorytm dzieli zawodników na `ILE_GRUP` grup tak, by w każdej byli do siebie
jak najbardziej podobni.

#### Metoda łokcia — skąd liczba grup

KMeans mierzy jakość **inercją** — sumą kwadratów odległości każdego zawodnika
od środka jego grupy. Im mniejsza, tym ciaśniejsze grupy.

Nie można po prostu wybrać najmniejszej, bo inercja **zawsze** maleje wraz
z liczbą grup. Przy liczbie grup równej liczbie zawodników spadłaby do zera —
każdy byłby sam. Idealny wynik, zerowa wartość poznawcza.

Dlatego rysujemy inercję dla k od 2 do 8 i patrzymy na **kształt**:
- na początku spadek stromy — każda nowa grupa realnie rozdziela różnych zawodników,
- potem wypłaszczenie — nowe grupy tylko kroją spójne skupiska.

Miejsce zgięcia to „łokieć". Wykres: `wykres_lokiec.png`.

**Uwaga:** skrypt nie wybiera k automatycznie. Ustawia `ILE_GRUP = 4` i rysuje
wykres do weryfikacji. Jeśli łokieć na Waszych danych wypada gdzie indziej,
zmieńcie tę liczbę — wtedy macie uzasadnienie oparte na wykresie.

#### Nazwy grup liczone z danych

Algorytm zwraca tylko numery. Nazwy nadajemy **my**, ale nie z góry — wyliczamy
je ze środków grup, na trzech osiach:

1. **Serwis** — czy średnia asów w grupie przekracza średnią wszystkich grup
2. **Return** — czy średnia asów traconych jest **poniżej** średniej
   (mniej traconych asów znaczy lepszy return)
3. **Stabilność** — podwójne błędy, używane tylko gdy dwie grupy wypadną
   identycznie na dwóch pierwszych osiach

Możliwe nazwy: **Serwis i return**, **Serwisanci**, **Returnerzy**,
**Uniwersalni**, a przy kolizji z dopiskiem *stabilny serwis* lub
*ryzykowny serwis*.

> Jeśli po przeliczeniu widać dużo nazw z myślnikiem, to sygnał, że grup jest
> o jedną za dużo. Warto wtedy wrócić do wykresu łokcia.

### Spojrzenie 2 — Role: kilka etykiet naraz

KMeans wrzuca każdego do jednej szufladki, a w rzeczywistości role się nakładają
— ktoś może świetnie serwować **i** świetnie returnować. Dlatego obok
klasteryzacji liczymy **role**, których zawodnik może mieć kilka albo żadnej.

Progi wyznaczamy **percentylami**, czyli relatywnie do stawki:

```python
prog_asy = tabela["asy"].quantile(0.75)
```

Percentyl 75 to wartość, poniżej której mieści się 75% zawodników. Kto ją
przekracza, należy do najlepszej ćwiartki. (To samo pojęcie co w wynikach
testów: „wynik w 90. percentylu" znaczy lepszy od 90% zdających.)

| Rola | Warunek |
|---|---|
| Serwisant | asy w górnych 25% |
| Mocny return | asy stracone w dolnych 25% |
| Stabilny serwis | podwójne błędy w dolnych 25% |
| Ryzykowny serwis | podwójne błędy w górnych 25% |
| Celny pierwszy serwis | % pierwszych serwisów w górnych 25% |
| Bez wyraźnych wyróżników | żaden z powyższych |

Zawodnik może mieć na przykład „Serwisant, Ryzykowny serwis" — dużo asów
okupionych podwójnymi błędami.

**Dlaczego percentyle, a nie średnia.** Przy progu na średniej mniej więcej
połowa stawki zostałaby serwisantami i etykieta przestałaby cokolwiek
wyróżniać. Górna ćwiartka daje węższą, sensowną grupę.

### Spojrzenie 3 — Gaussian Mixture: przynależność procentowa

**GaussianMixture** to „miękka" wersja KMeansa. Zamiast „jesteś w grupie 2"
mówi „jesteś w 65% serwisantem, w 30% uniwersalnym".

Dzięki temu widać zawodników stojących w rozkroku między stylami. Skrypt
wypisuje tych, u których główny styl nie przekracza **70%** przynależności —
to właśnie przypadki, których jedna etykieta nie opisuje.

Składniki GMM nazywamy tą samą funkcją co grupy KMeans, żeby nazewnictwo było
spójne.

### Co zapisuje

`style_gry.csv` z kolumnami: cztery cechy, `mecze`, `grupa`, `styl`, `role`,
`styl_glowny`, `styl_glowny_proc`, `styl_drugi`, `styl_drugi_proc`.

Wykresy: `wykres_style.png` (asy kontra asy stracone, kolory to grupy)
i `wykres_lokiec.png`.

---

## 10. Dodatek — `top10.py`

Niezależny od reszty, czyta prosto z `dane/`. Dla każdej nawierzchni pokazuje
**dwie** listy:

1. TOP 10 według **liczby wygranych**
2. TOP 10 według **skuteczności procentowej**, z progiem minimum 20 meczów

**Po co dwie.** Sama liczba wygranych premiuje tych, którzy grali najwięcej
turniejów. Skuteczność to wyrównuje, ale bez progu minimum na szczycie
wylądowałby ktoś, kto zagrał trzy mecze i wygrał trzy. Próg zmienia się
w zmiennej `minimum`.

---

## 11. Dodatek — `porownaj.py`

Wersja terminalowa porównania dwóch zawodników. Wpisujesz fragment nazwiska,
dostajesz bilans H2H, statystyki, Elo i szansę na wygraną wyliczoną ze wzoru Elo.

Służy jako plan awaryjny, gdyby Streamlit nie wstał na obronie.

---

## 12. Aplikacja — `app.py`

Interfejs w Streamlit. Na starcie sprawdza, czy wszystkie potrzebne pliki
istnieją, i jeśli nie — wypisuje, co uruchomić.

### Sekcja główna

Wybierasz dwóch zawodników i nawierzchnię. Dla każdego pokazuje:

| Element | Skąd pochodzi |
|---|---|
| Szansa na wygraną | `model_wygrana.pkl`, metoda `predict_proba` |
| Ranking ATP | z ostatniego meczu, w którym był zapisany |
| Elo na nawierzchni | `ranking_elo_nawierzchnie.csv` |
| Forma | wygrane z ostatnich 10 meczów |
| Prognoza asów | `model_asy.pkl`, przeliczona na typową liczbę punktów serwisowych |
| Role | `style_gry.csv` |
| Przynależność procentowa | wynik GaussianMixture |

Pod spodem tabela porównująca **dwie metody liczenia szansy**: model
i sam wzór Elo. Różnica bierze się stąd, że model widzi dodatkowo ranking ATP,
wzrost i nawierzchnię.

### Obsługa braku danych

Jeśli zawodnik nie grał na wybranej nawierzchni, Elo pokazuje **„brak danych"**,
a nie wartość 1500. To ważne — 1500 to wartość startowa, nie wynik, a część
zawodników faktycznie ma Elo w okolicach 1500 po rozegranych meczach. Mylenie
tych dwóch rzeczy fałszowałoby obraz.

Przy mniej niż 5 meczach (`MALO_MECZOW`) wartość jest pokazywana z ostrzeżeniem.
Jeśli któregoś z wybranych zawodników to dotyczy, nad prognozą pojawia się
komunikat, że model podstawia wartość startową i prognoza jest niepewna.

### Zakładki

**Profil zawodnika** — uśrednione statystyki w rozbiciu: ogółem i osobno dla
każdej nawierzchni. Kolumny „w meczu" to surowe średnie, kolumny „/100 pkt"
to wartości znormalizowane.

**Ostatnie mecze** — 10 ostatnich spotkań z datą, turniejem, przeciwnikiem,
wynikiem oraz asami i błędami w obu wersjach.

**Bezpośrednie mecze** — bilans wg nawierzchni, średnie z meczów między tą parą,
porównanie „jak gra przeciwko temu rywalowi kontra jak gra ogólnie" oraz lista
wszystkich wspólnych meczów z liczbą asów każdego.

**Statystyki** — porównanie, wykres słupkowy, asy w 30 ostatnich meczach oraz
wykres punktowy zależności między procentem pierwszego serwisu a asami wraz
ze współczynnikiem korelacji.

**Style gry** — role, zawodnicy na granicy stylów, podział KMeans, wykresy.

**Rankingi** — TOP 20 Elo ogólne i na wybranej nawierzchni.

---

## 13. Ograniczenia

### Czego nie ma w danych

- **kierunku serwisu** (na „T", w środek, szeroko),
- **rodzaju serwisu** (płaski, slajs, kick),
- **ustawienia returnera** (blisko linii czy daleko za nią),
- **podziału hala / kort otwarty** — jest tylko rodzaj nawierzchni,
- **warunków pogodowych**,
- **informacji o kontuzjach i zmęczeniu**.

Dane o pozycjach zawodników pochodzą z systemów śledzących typu Hawk-Eye i są
własnością turniejów — nie ma ich w otwartych zbiorach. Częściowe informacje
o rodzaju serwisu zawiera *Match Charting Project*, ale obejmuje tylko ułamek
meczów i ma inną strukturę.

### Czego nie robi model

**Bilans bezpośrednich meczów nie wchodzi do modelu.** Aplikacja go wyświetla,
ale model go nie używa. Powody: większość par zagrała ze sobą 0–3 razy, więc
cecha byłaby w większości przypadków zerem, a policzenie jej z całego zbioru
groziłoby wyciekiem danych. Dałoby się to zrobić poprawnie — licząc bilans
tylko z wcześniejszych meczów, tak jak przy Elo — ale nie zostało wdrożone.

**Styl gry to interpretacja, nie pomiar.** Model może wykryć, że pewne grupy
zawodników mają podobne statystyki. Nie udowodni, że ktoś *stoi przy bandzie*
ani *serwuje slajsem*. W raporcie należy to nazywać „stylem gry wywnioskowanym
ze statystyk".

**Prognoza to nie pewnik.** Model nie uwzględnia formy dziennej, kontuzji,
zmęczenia po poprzednim meczu ani warunków. Nie jest narzędziem do obstawiania.

---

## 14. Słowniczek pojęć

| Pojęcie | Znaczenie |
|---|---|
| **Normalizacja** | sprowadzenie wartości do wspólnego mianownika, u nas: na 100 punktów serwisowych |
| **Percentyl** | wartość, poniżej której mieści się dany procent obserwacji |
| **Korelacja** | miara zależności liniowej, od −1 do 1; nie dowodzi przyczyny |
| **Wyciek danych** | sytuacja, gdy model dostaje informację z przyszłości |
| **Klasyfikacja** | przewidywanie kategorii (kto wygra) |
| **Regresja** | przewidywanie liczby (ile asów) |
| **Uczenie nienadzorowane** | szukanie struktury bez podanych etykiet |
| **Kodowanie one-hot** | zamiana tekstu na kolumny 0/1 |
| **Inercja** | suma kwadratów odległości od środków grup w KMeans |
| **Dokładność** | procent trafnych przewidywań klasyfikatora |
| **MAE** | średni błąd bezwzględny w regresji |
| **R²** | ile zmienności model wyjaśnia; 0 to tyle co zgadywanie średniej |
| **Macierz pomyłek** | tabela trafień i pomyłek dla każdej klasy |

---

## 15. Użyte funkcje Pandas

| Funkcja | Gdzie | Do czego |
|---|---|---|
| `read_csv()` | wszędzie | wczytanie danych |
| `concat()` | projekt.py, style.py | sklejanie tabel |
| `to_datetime()` | projekt.py | konwersja daty |
| `sort_values()` | projekt.py, elo.py | sortowanie chronologiczne |
| `dropna()` | projekt.py, model.py | usuwanie braków |
| `fillna()` | model_asy.py | uzupełnianie braków średnią |
| `str.contains()` | projekt.py | wykrywanie kreczów |
| `str.endswith()` | elo.py, app.py | filtrowanie po nawierzchni |
| `value_counts()` | eda.py, top10.py | zliczanie wystąpień |
| `describe()` | projekt.py | statystyki opisowe |
| `groupby()` | eda.py, style.py | grupowanie |
| `mean()` | wszędzie | średnie |
| `quantile()` | style.py | progi percentylowe |
| `corr()` | eda.py, app.py | korelacje |
| `unstack()` | eda.py | zamiana grupowania w tabelę |
| `add()` | top10.py | sumowanie serii z `fill_value` |
| `map()` | style.py, model_asy.py | podstawianie wartości ze słownika |
| `iloc[]` | elo.py, model.py | dostęp po numerze wiersza |
| `astype()` | projekt.py, app.py | zmiana typu |
| `round()` | wszędzie | zaokrąglanie |
| `head()` / `tail()` | wszędzie | pierwsze / ostatnie wiersze |
| `to_csv()` | projekt.py, elo.py, style.py | zapis wyników |

## 16. Użyte narzędzia scikit-learn

| Klasa | Plik | Rola |
|---|---|---|
| `LogisticRegression` | model.py | klasyfikacja zwycięzcy |
| `LinearRegression` | model_asy.py | regresja liczby asów |
| `KMeans` | style.py | klasteryzacja twarda |
| `GaussianMixture` | style.py | klasteryzacja miękka |
| `StandardScaler` | style.py | skalowanie cech |
| `accuracy_score`, `confusion_matrix` | model.py | ocena klasyfikacji |
| `mean_absolute_error`, `r2_score` | model_asy.py | ocena regresji |

---

## 17. Źródło danych i licencja

Dane: **Jeff Sackmann**, zbiór `tennis_atp`, przez mirror archiwalny
`Aneeshers/tennis-sackmann-archive`.

Licencja: **CC BY-NC-SA 4.0** — użycie niekomercyjne, z podaniem autora,
na tej samej licencji. Projekt studencki mieści się w jej warunkach.
W bibliografii raportu należy wymienić Jeffa Sackmanna jako autora danych.
