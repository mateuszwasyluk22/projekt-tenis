import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# --- USTAWIENIA STRONY ---

st.set_page_config(page_title="Analiza meczów ATP", layout="wide")

st.title("Analiza i prognoza meczów ATP")
st.caption("Projekt zaliczeniowy – Systemy Sztucznej Inteligencji")


# --- SPRAWDZENIE CZY WSZYSTKO JEST POLICZONE ---

potrzebne_pliki = ["wyniki/dane_z_elo.csv", "wyniki/ranking_elo.csv",
                   "wyniki/ranking_elo_nawierzchnie.csv", "wyniki/model_wygrana.pkl",
                   "wyniki/model_asy.pkl", "wyniki/style_gry.csv"]

brakuje = []
for p in potrzebne_pliki:
    if not os.path.exists(p):
        brakuje.append(p)

if len(brakuje) > 0:
    st.error("Brakuje plików: " + ", ".join(brakuje))
    st.write("Uruchom po kolei:")
    st.code("python projekt.py\npython elo.py\npython model.py\npython model_asy.py\npython style.py")
    st.stop()


# --- WCZYTANIE DANYCH ---

@st.cache_data
def wczytaj():
    df = pd.read_csv("wyniki/dane_z_elo.csv")
    df["data"] = pd.to_datetime(df["data"])
    ranking = pd.read_csv("wyniki/ranking_elo.csv", index_col=0)
    ranking_naw = pd.read_csv("wyniki/ranking_elo_nawierzchnie.csv", index_col=0)
    style = pd.read_csv("wyniki/style_gry.csv", index_col=0)
    return df, ranking, ranking_naw, style


@st.cache_resource
def wczytaj_modele():
    plik = open("wyniki/model_wygrana.pkl", "rb")
    model_wygrana = pickle.load(plik)
    plik.close()

    plik = open("wyniki/model_asy.pkl", "rb")
    model_asy = pickle.load(plik)
    plik.close()

    return model_wygrana, model_asy


df, ranking, ranking_naw, style = wczytaj()
model_wygrana, model_asy = wczytaj_modele()

wszyscy = sorted(pd.concat([df["winner_name"], df["loser_name"]]).unique())


# --- WSZYSTKIE MECZE JEDNEGO ZAWODNIKA ---

@st.cache_data
def mecze_gracza(gracz):
    jako_zw = df[df["winner_name"] == gracz]
    jako_prz = df[df["loser_name"] == gracz]

    a = pd.DataFrame({
        "data": jako_zw["data"],
        "turniej": jako_zw["tourney_name"],
        "nawierzchnia": jako_zw["surface"],
        "przeciwnik": jako_zw["loser_name"],
        "wynik": "W",
        "wygrana": 1,
        "asy_surowe": jako_zw["w_ace"],
        "bledy_surowe": jako_zw["w_df"],
        "asy": jako_zw["asy_zwyciezca"],
        "asy_stracone": jako_zw["asy_przegrany"],
        "bledy": jako_zw["bledy_zwyciezca"],
        "pierwszy_serwis": jako_zw["pierwszy_zwyciezca"],
        "punkty_serwisowe": jako_zw["w_svpt"],
        "ranking_atp": jako_zw["winner_rank"],
        "wzrost": jako_zw["winner_ht"],
        "reka": jako_zw["winner_hand"]
    })

    b = pd.DataFrame({
        "data": jako_prz["data"],
        "turniej": jako_prz["tourney_name"],
        "nawierzchnia": jako_prz["surface"],
        "przeciwnik": jako_prz["winner_name"],
        "wynik": "P",
        "wygrana": 0,
        "asy_surowe": jako_prz["l_ace"],
        "bledy_surowe": jako_prz["l_df"],
        "asy": jako_prz["asy_przegrany"],
        "asy_stracone": jako_prz["asy_zwyciezca"],
        "bledy": jako_prz["bledy_przegrany"],
        "pierwszy_serwis": jako_prz["pierwszy_przegrany"],
        "punkty_serwisowe": jako_prz["l_svpt"],
        "ranking_atp": jako_prz["loser_rank"],
        "wzrost": jako_prz["loser_ht"],
        "reka": jako_prz["loser_hand"]
    })

    return pd.concat([a, b]).sort_values("data")


# --- PODSTAWOWE INFORMACJE O ZAWODNIKU ---

def opis_gracza(gracz):
    mecze = mecze_gracza(gracz)

    wzrost = mecze["wzrost"].dropna()
    reka = mecze["reka"].dropna()
    pozycje = mecze["ranking_atp"].dropna()

    # styl gry z klasteryzacji, jesli zawodnik ma dosc meczow
    if gracz in style.index:
        styl = style.loc[gracz, "styl"]
        role = style.loc[gracz, "role"]
        styl_glowny = style.loc[gracz, "styl_glowny"]
        styl_glowny_proc = style.loc[gracz, "styl_glowny_proc"]
        styl_drugi = style.loc[gracz, "styl_drugi"]
        styl_drugi_proc = style.loc[gracz, "styl_drugi_proc"]
    else:
        styl = "za mało meczów"
        role = "za mało meczów"
        styl_glowny = None
        styl_glowny_proc = None
        styl_drugi = None
        styl_drugi_proc = None

    return {
        "wzrost": wzrost.iloc[-1] if len(wzrost) > 0 else 185,
        "reka": reka.iloc[-1] if len(reka) > 0 else "R",
        "ranking": int(pozycje.iloc[-1]) if len(pozycje) > 0 else None,
        "styl": styl,
        "role": role,
        "styl_glowny": styl_glowny,
        "styl_glowny_proc": styl_glowny_proc,
        "styl_drugi": styl_drugi,
        "styl_drugi_proc": styl_drugi_proc,
        "punkty_serwisowe": mecze["punkty_serwisowe"].mean(),
        "pierwszy_serwis": mecze["pierwszy_serwis"].mean(),
        "wygrane": int(mecze.tail(10)["wygrana"].sum()),
        "z_ilu": len(mecze.tail(10))
    }


# --- ELO ---

def elo_ogolne(gracz):
    if gracz in ranking.index:
        return ranking.loc[gracz, "elo"]
    return 1500


def elo_nawierzchnia(gracz, nawierzchnia):
    klucz = gracz + "_" + nawierzchnia
    if klucz in ranking_naw.index:
        return ranking_naw.loc[klucz, "elo"]
    return 1500


# ile meczow zawodnik rozegral na danej nawierzchni
def ile_meczow(gracz, nawierzchnia):
    mecze = mecze_gracza(gracz)
    return len(mecze[mecze["nawierzchnia"] == nawierzchnia])


# czy w ogole mamy dla niego Elo na tej nawierzchni
def ma_elo(gracz, nawierzchnia):
    return (gracz + "_" + nawierzchnia) in ranking_naw.index


# ponizej tylu meczow traktujemy Elo jako niepewne
MALO_MECZOW = 5


# --- WYBÓR MECZU ---

st.header("Wybierz mecz")

kol1, kol2, kol3 = st.columns(3)

with kol1:
    gracz_a = st.selectbox("Zawodnik A", wszyscy, index=0)
with kol2:
    gracz_b = st.selectbox("Zawodnik B", wszyscy, index=1)
with kol3:
    nawierzchnia = st.selectbox("Nawierzchnia", ["Hard", "Clay", "Grass"])

if gracz_a == gracz_b:
    st.warning("Wybierz dwóch różnych zawodników.")
    st.stop()

info_a = opis_gracza(gracz_a)
info_b = opis_gracza(gracz_b)


# --- PROGNOZA Z MODELU KLASYFIKACJI ---

def szansa_z_modelu(gracz_1, gracz_2):
    # budujemy jeden wiersz z takimi samymi cechami jak przy uczeniu
    wiersz = {
        "roznica_elo": elo_ogolne(gracz_1) - elo_ogolne(gracz_2),
        "roznica_elo_naw": elo_nawierzchnia(gracz_1, nawierzchnia) - elo_nawierzchnia(gracz_2, nawierzchnia),
        "roznica_rank": 0,
        "roznica_wzrost": opis_gracza(gracz_1)["wzrost"] - opis_gracza(gracz_2)["wzrost"],
        "twarda": 1 if nawierzchnia == "Hard" else 0,
        "ziemia": 1 if nawierzchnia == "Clay" else 0,
        "trawa": 1 if nawierzchnia == "Grass" else 0
    }

    # roznica rankingow ATP, jesli obaj je maja
    r1 = opis_gracza(gracz_1)["ranking"]
    r2 = opis_gracza(gracz_2)["ranking"]
    if r1 is not None and r2 is not None:
        wiersz["roznica_rank"] = r2 - r1

    X = pd.DataFrame([wiersz])[model_wygrana["cechy"]]
    return model_wygrana["model"].predict_proba(X)[0][1]


# --- PROGNOZA Z MODELU REGRESJI ---

def asy_z_modelu(serwujacy, przeciwnik):
    info_s = opis_gracza(serwujacy)
    info_p = opis_gracza(przeciwnik)

    srednia = model_asy["srednia_ogolna"]

    wiersz = {
        "serwis_historia": model_asy["serwis_gracza"].get(serwujacy, srednia),
        "straty_przeciwnika": model_asy["straty_gracza"].get(przeciwnik, srednia),
        "wzrost_serwujacego": info_s["wzrost"],
        "wzrost_przeciwnika": info_p["wzrost"],
        "przeciwnik_leworeczny": 1 if info_p["reka"] == "L" else 0,
        "ranking_przeciwnika": info_p["ranking"] if info_p["ranking"] is not None else 100,
        "twarda": 1 if nawierzchnia == "Hard" else 0,
        "ziemia": 1 if nawierzchnia == "Clay" else 0,
        "trawa": 1 if nawierzchnia == "Grass" else 0
    }

    X = pd.DataFrame([wiersz])[model_asy["cechy"]]
    na_sto = model_asy["model"].predict(X)[0]

    # przeliczenie na caly mecz
    w_meczu = na_sto / 100 * info_s["punkty_serwisowe"]

    return na_sto, w_meczu


# --- OBLICZENIA ---

szansa_model_a = szansa_z_modelu(gracz_a, gracz_b)
szansa_model_b = 1 - szansa_model_a

elo_a = elo_nawierzchnia(gracz_a, nawierzchnia)
elo_b = elo_nawierzchnia(gracz_b, nawierzchnia)
szansa_elo_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

asy_sto_a, asy_mecz_a = asy_z_modelu(gracz_a, gracz_b)
asy_sto_b, asy_mecz_b = asy_z_modelu(gracz_b, gracz_a)


# --- PROGNOZA ---

st.header("Prognoza")

kol1, kol2 = st.columns(2)

# wypisuje Elo albo informacje ze danych brakuje
def pokaz_elo(gracz, wartosc):
    ile = ile_meczow(gracz, nawierzchnia)

    if not ma_elo(gracz, nawierzchnia):
        st.metric("Elo na nawierzchni " + nawierzchnia, "brak danych")
        st.caption("Nie rozegrał żadnego meczu na tej nawierzchni w analizowanym okresie.")
    elif ile < MALO_MECZOW:
        st.metric("Elo na nawierzchni " + nawierzchnia, round(wartosc))
        st.caption("Uwaga: tylko " + str(ile) + " meczów na tej nawierzchni, wartość mało pewna.")
    else:
        st.metric("Elo na nawierzchni " + nawierzchnia, round(wartosc))
        st.caption("Na podstawie " + str(ile) + " meczów.")


with kol1:
    st.subheader(gracz_a)
    st.metric("Szansa na wygraną (model)", str(round(szansa_model_a * 100, 1)) + " %")
    if info_a["ranking"] is not None:
        st.metric("Ranking ATP", info_a["ranking"])
    pokaz_elo(gracz_a, elo_a)
    st.metric("Forma", str(info_a["wygrane"]) + " / " + str(info_a["z_ilu"]) + " ostatnich")
    st.metric("Prognoza asów w meczu", round(asy_mecz_a, 1))
    st.write("**Role:** " + str(info_a["role"]))
    if info_a["styl_glowny"] is not None:
        st.caption("Przynależność: " + info_a["styl_glowny"] + " "
                   + str(info_a["styl_glowny_proc"]) + "%, "
                   + info_a["styl_drugi"] + " " + str(info_a["styl_drugi_proc"]) + "%")

with kol2:
    st.subheader(gracz_b)
    st.metric("Szansa na wygraną (model)", str(round(szansa_model_b * 100, 1)) + " %")
    if info_b["ranking"] is not None:
        st.metric("Ranking ATP", info_b["ranking"])
    pokaz_elo(gracz_b, elo_b)
    st.metric("Forma", str(info_b["wygrane"]) + " / " + str(info_b["z_ilu"]) + " ostatnich")
    st.metric("Prognoza asów w meczu", round(asy_mecz_b, 1))
    st.write("**Role:** " + str(info_b["role"]))
    if info_b["styl_glowny"] is not None:
        st.caption("Przynależność: " + info_b["styl_glowny"] + " "
                   + str(info_b["styl_glowny_proc"]) + "%, "
                   + info_b["styl_drugi"] + " " + str(info_b["styl_drugi_proc"]) + "%")

# ostrzezenie gdy prognoza stoi na slabych danych
niepewni = []
for gracz in [gracz_a, gracz_b]:
    if not ma_elo(gracz, nawierzchnia) or ile_meczow(gracz, nawierzchnia) < MALO_MECZOW:
        niepewni.append(gracz)

if len(niepewni) > 0:
    st.warning("Mało danych na nawierzchni " + nawierzchnia + " dla: " + ", ".join(niepewni) +
               ". Tam, gdzie Elo nie istnieje, model podstawia wartość startową 1500, "
               "więc prognoza jest obarczona dużą niepewnością.")

st.progress(szansa_model_a)
st.caption("Pasek pokazuje szansę zawodnika A według modelu")

# porownanie dwoch sposobow liczenia
st.write("**Porównanie metod**")
porownanie_metod = pd.DataFrame({
    "Metoda": ["Regresja logistyczna (model)", "Sam wzór Elo"],
    gracz_a: [str(round(szansa_model_a * 100, 1)) + " %", str(round(szansa_elo_a * 100, 1)) + " %"],
    gracz_b: [str(round(szansa_model_b * 100, 1)) + " %", str(round((1 - szansa_elo_a) * 100, 1)) + " %"]
})
st.dataframe(porownanie_metod, hide_index=True)

st.caption("Model uwzględnia różnicę Elo, Elo na nawierzchni, ranking ATP, różnicę wzrostu "
           "i rodzaj nawierzchni. Wzór Elo patrzy tylko na różnicę punktów.")


# --- ZAKŁADKI ---

z0, z1, z2, z3, z4, z5 = st.tabs(
    ["Profil zawodnika", "Ostatnie mecze", "Bezpośrednie mecze", "Statystyki",
     "Style gry", "Rankingi"])


# ZAKŁADKA 0: PROFIL ZAWODNIKA
with z0:
    st.subheader("Uśrednione statystyki z całej kariery w zbiorze")

    # liczy srednie dla jednego zawodnika, osobno dla kazdej nawierzchni
    def profil(gracz):
        mecze = mecze_gracza(gracz)
        wiersze = []

        # najpierw wszystko razem, potem kazda nawierzchnia osobno
        grupy = [("Ogółem", mecze)]
        for nazwa in ["Hard", "Clay", "Grass"]:
            grupy.append((nazwa, mecze[mecze["nawierzchnia"] == nazwa]))

        for nazwa, czesc in grupy:
            if len(czesc) == 0:
                wiersze.append({
                    "Zakres": nazwa, "Mecze": 0, "Wygrane": 0, "Skuteczność %": None,
                    "Asy /100 pkt": None, "Asy w meczu": None,
                    "Asy stracone /100 pkt": None, "Podwójne błędy /100 pkt": None,
                    "Podwójne błędy w meczu": None, "Pierwszy serwis %": None
                })
                continue

            wiersze.append({
                "Zakres": nazwa,
                "Mecze": len(czesc),
                "Wygrane": int(czesc["wygrana"].sum()),
                "Skuteczność %": round(czesc["wygrana"].mean() * 100, 1),
                "Asy /100 pkt": round(czesc["asy"].mean(), 2),
                "Asy w meczu": round(czesc["asy_surowe"].mean(), 1),
                "Asy stracone /100 pkt": round(czesc["asy_stracone"].mean(), 2),
                "Podwójne błędy /100 pkt": round(czesc["bledy"].mean(), 2),
                "Podwójne błędy w meczu": round(czesc["bledy_surowe"].mean(), 1),
                "Pierwszy serwis %": round(czesc["pierwszy_serwis"].mean(), 1)
            })

        return pd.DataFrame(wiersze)

    for gracz in [gracz_a, gracz_b]:
        info = opis_gracza(gracz)

        naglowek = gracz + " – " + str(int(info["wzrost"])) + " cm, "
        naglowek = naglowek + ("leworęczny" if info["reka"] == "L" else "praworęczny")
        if info["ranking"] is not None:
            naglowek = naglowek + ", ranking ATP " + str(info["ranking"])
        naglowek = naglowek + ", role: " + str(info["role"])

        st.write("**" + naglowek + "**")
        st.dataframe(profil(gracz), hide_index=True)

    st.caption("Puste pola oznaczają, że zawodnik nie rozegrał meczu na tej nawierzchni "
               "w analizowanym okresie. Kolumny „w meczu” to surowe średnie z pojedynczego "
               "meczu, kolumny „/100 pkt” to te same wartości przeliczone na 100 punktów "
               "serwisowych, żeby mecze różnej długości dało się porównywać.")


# ZAKŁADKA 1: FORMA
with z1:
    st.subheader("Ostatnie 10 meczów")

    for gracz in [gracz_a, gracz_b]:
        st.write("**" + gracz + "**")

        ostatnie = mecze_gracza(gracz).tail(10).iloc[::-1]

        pokaz = pd.DataFrame({
            "Data": ostatnie["data"].dt.date,
            "Turniej": ostatnie["turniej"],
            "Nawierzchnia": ostatnie["nawierzchnia"],
            "Przeciwnik": ostatnie["przeciwnik"],
            "Wynik": ostatnie["wynik"],
            "Asy": ostatnie["asy_surowe"].astype(int),
            "Podwójne błędy": ostatnie["bledy_surowe"].astype(int),
            "Asy /100 pkt": ostatnie["asy"].round(1),
            "Błędy /100 pkt": ostatnie["bledy"].round(1)
        })

        st.dataframe(pokaz, hide_index=True)

    st.caption("Kolumny „Asy” i „Podwójne błędy” to liczby z danego meczu. "
               "Kolumny „/100 pkt” przeliczają je na 100 punktów serwisowych, "
               "żeby mecze o różnej długości dało się porównywać.")


# ZAKŁADKA 2: BEZPOŚREDNIE MECZE
with z2:
    h2h_a = df[(df["winner_name"] == gracz_a) & (df["loser_name"] == gracz_b)]
    h2h_b = df[(df["winner_name"] == gracz_b) & (df["loser_name"] == gracz_a)]

    if len(h2h_a) + len(h2h_b) == 0:
        st.write("Ci zawodnicy nie grali ze sobą w analizowanym okresie.")
    else:
        st.subheader("Bilans: " + str(len(h2h_a)) + " – " + str(len(h2h_b)))

        wiersze = []
        for nazwa in ["Hard", "Clay", "Grass"]:
            wiersze.append({
                "Nawierzchnia": nazwa,
                gracz_a: len(h2h_a[h2h_a["surface"] == nazwa]),
                gracz_b: len(h2h_b[h2h_b["surface"] == nazwa])
            })
        st.dataframe(pd.DataFrame(wiersze), hide_index=True)

        # --- STATYSTYKI TYLKO Z MECZOW MIEDZY TA DWOJKA ---
        # W h2h_a zawodnik A jest zwyciezca, wiec jego dane sa w kolumnach w_.
        # W h2h_b jest przegranym, wiec jego dane sa w kolumnach l_. Odwracamy.

        czesc_a = pd.DataFrame({
            "data": h2h_a["data"],
            "turniej": h2h_a["tourney_name"],
            "nawierzchnia": h2h_a["surface"],
            "wygral": h2h_a["winner_name"],
            "wynik": h2h_a["score"],
            "asy_a_surowe": h2h_a["w_ace"],
            "asy_b_surowe": h2h_a["l_ace"],
            "asy_a": h2h_a["asy_zwyciezca"],
            "asy_b": h2h_a["asy_przegrany"],
            "bledy_a": h2h_a["bledy_zwyciezca"],
            "bledy_b": h2h_a["bledy_przegrany"],
            "pierwszy_a": h2h_a["pierwszy_zwyciezca"],
            "pierwszy_b": h2h_a["pierwszy_przegrany"]
        })

        czesc_b = pd.DataFrame({
            "data": h2h_b["data"],
            "turniej": h2h_b["tourney_name"],
            "nawierzchnia": h2h_b["surface"],
            "wygral": h2h_b["winner_name"],
            "wynik": h2h_b["score"],
            "asy_a_surowe": h2h_b["l_ace"],
            "asy_b_surowe": h2h_b["w_ace"],
            "asy_a": h2h_b["asy_przegrany"],
            "asy_b": h2h_b["asy_zwyciezca"],
            "bledy_a": h2h_b["bledy_przegrany"],
            "bledy_b": h2h_b["bledy_zwyciezca"],
            "pierwszy_a": h2h_b["pierwszy_przegrany"],
            "pierwszy_b": h2h_b["pierwszy_zwyciezca"]
        })

        wszystkie = pd.concat([czesc_a, czesc_b]).sort_values("data")

        st.subheader("Średnie w meczach między sobą")

        srednie_h2h = pd.DataFrame({
            gracz_a: {
                "Asy /100 pkt": round(wszystkie["asy_a"].mean(), 2),
                "Podwójne błędy /100 pkt": round(wszystkie["bledy_a"].mean(), 2),
                "Pierwszy serwis %": round(wszystkie["pierwszy_a"].mean(), 1),
                "Asy średnio w meczu": round(wszystkie["asy_a_surowe"].mean(), 1)
            },
            gracz_b: {
                "Asy /100 pkt": round(wszystkie["asy_b"].mean(), 2),
                "Podwójne błędy /100 pkt": round(wszystkie["bledy_b"].mean(), 2),
                "Pierwszy serwis %": round(wszystkie["pierwszy_b"].mean(), 1),
                "Asy średnio w meczu": round(wszystkie["asy_b_surowe"].mean(), 1)
            }
        })
        st.dataframe(srednie_h2h)

        # porownanie: jak graja ze soba kontra jak graja ogolnie
        st.subheader("Mecze między sobą a reszta sezonu")

        porownanie_h2h = []
        for gracz, kolumna in [(gracz_a, "asy_a"), (gracz_b, "asy_b")]:
            ogolem = mecze_gracza(gracz)["asy"].mean()
            w_h2h = wszystkie[kolumna].mean()
            porownanie_h2h.append({
                "Zawodnik": gracz,
                "Asy /100 pkt ogółem": round(ogolem, 2),
                "Asy /100 pkt w H2H": round(w_h2h, 2),
                "Różnica": round(w_h2h - ogolem, 2)
            })
        st.dataframe(pd.DataFrame(porownanie_h2h), hide_index=True)

        st.caption("Dodatnia różnica oznacza, że przeciwko temu rywalowi zawodnik "
                   "serwuje lepiej niż zwykle.")

        st.subheader("Wszystkie mecze")

        st.dataframe(pd.DataFrame({
            "Data": wszystkie["data"].dt.date,
            "Turniej": wszystkie["turniej"],
            "Nawierzchnia": wszystkie["nawierzchnia"],
            "Wygrał": wszystkie["wygral"],
            "Wynik": wszystkie["wynik"],
            "Asy " + gracz_a: wszystkie["asy_a_surowe"].astype(int),
            "Asy " + gracz_b: wszystkie["asy_b_surowe"].astype(int)
        }).iloc[::-1], hide_index=True)


# ZAKŁADKA 3: STATYSTYKI
with z3:
    def statystyki(gracz):
        mecze = mecze_gracza(gracz)
        na_naw = mecze[mecze["nawierzchnia"] == nawierzchnia]

        if len(na_naw) > 0:
            skutecznosc_naw = round(na_naw["wygrana"].mean() * 100, 1)
        else:
            skutecznosc_naw = 0

        return {
            "Mecze razem": len(mecze),
            "Wygrane": int(mecze["wygrana"].sum()),
            "Skuteczność %": round(mecze["wygrana"].mean() * 100, 1),
            "Mecze na " + nawierzchnia: len(na_naw),
            "Skuteczność na " + nawierzchnia + " %": skutecznosc_naw,
            "Asy /100 pkt": round(mecze["asy"].mean(), 2),
            "Asy stracone /100 pkt": round(mecze["asy_stracone"].mean(), 2),
            "Podwójne błędy /100 pkt": round(mecze["bledy"].mean(), 2),
            "Pierwszy serwis %": round(mecze["pierwszy_serwis"].mean(), 1)
        }

    porownanie = pd.DataFrame({gracz_a: statystyki(gracz_a), gracz_b: statystyki(gracz_b)})
    st.dataframe(porownanie)

    do_wykresu = porownanie.loc[["Asy /100 pkt", "Asy stracone /100 pkt",
                                 "Podwójne błędy /100 pkt"]]

    rysunek, os1 = plt.subplots(figsize=(8, 4))
    do_wykresu.plot(kind="bar", ax=os1)
    os1.set_ylabel("Średnia na 100 punktów serwisowych")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(rysunek)

    st.subheader("Asy w kolejnych meczach")

    rysunek2, os2 = plt.subplots(figsize=(10, 4))
    for gracz in [gracz_a, gracz_b]:
        mecze = mecze_gracza(gracz).tail(30)
        os2.plot(mecze["data"], mecze["asy"], marker="o", label=gracz)
    os2.set_ylabel("Asy na 100 punktów")
    os2.legend()
    os2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(rysunek2)

    # --- CZY PIERWSZY SERWIS WIAZE SIE Z ASAMI ---

    st.subheader("Pierwszy serwis a asy")

    rysunek3, os3 = plt.subplots(figsize=(9, 5))

    korelacje_graczy = []

    for gracz in [gracz_a, gracz_b]:
        mecze = mecze_gracza(gracz)
        os3.scatter(mecze["pierwszy_serwis"], mecze["asy"], alpha=0.5, s=25, label=gracz)

        # korelacja miedzy procentem pierwszego serwisu a asami
        wspolczynnik = mecze["pierwszy_serwis"].corr(mecze["asy"])
        korelacje_graczy.append({
            "Zawodnik": gracz,
            "Mecze": len(mecze),
            "Korelacja pierwszy serwis – asy": round(wspolczynnik, 3)
        })

    os3.set_xlabel("Procent trafionych pierwszych serwisów")
    os3.set_ylabel("Asy na 100 punktów")
    os3.legend()
    os3.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(rysunek3)

    st.dataframe(pd.DataFrame(korelacje_graczy), hide_index=True)

    # korelacja dla calego zbioru, jako punkt odniesienia
    wszystko_pierwszy = pd.concat([df["pierwszy_zwyciezca"], df["pierwszy_przegrany"]])
    wszystko_asy = pd.concat([df["asy_zwyciezca"], df["asy_przegrany"]])
    ogolna = wszystko_pierwszy.corr(wszystko_asy)

    st.caption("Dla całego zbioru (" + str(len(wszystko_asy)) + " wyników) korelacja wynosi "
               + str(round(ogolna, 3)) + ". Wartość bliska zeru oznacza brak związku, "
               "dodatnia – że wyższy procent pierwszych serwisów idzie w parze z większą "
               "liczbą asów. Korelacja nie dowodzi przyczyny: oba mogą wynikać z tego, "
               "że zawodnik miał dobry dzień serwisowy.")


# ZAKŁADKA 4: STYLE GRY
with z4:
    st.subheader("Style gry wykryte przez algorytm")

    st.write("Trzy spojrzenia na to samo. **KMeans** przypisuje każdego do jednej grupy. "
             "**Role** pozwalają mieć kilka etykiet naraz – ktoś może być jednocześnie "
             "serwisantem i dobrym returnerem. **Gaussian Mixture** podaje przynależność "
             "procentową, więc widać zawodników stojących w rozkroku między stylami.")

    st.subheader("Role – zawodnik może mieć kilka")

    st.write("Role przyznajemy po percentylach: górne 25% w asach to serwisant, "
             "dolne 25% w asach straconych to mocny return i tak dalej. "
             "Progi liczone są z rozkładu w naszym zbiorze, a nie z góry ustalone.")

    st.dataframe(style["role"].value_counts().head(15))

    st.subheader("Na granicy dwóch stylów")

    na_granicy = style[style["styl_glowny_proc"] < 70].sort_values("styl_glowny_proc")

    if len(na_granicy) == 0:
        st.write("Każdy zawodnik należy wyraźnie do jednej grupy (powyżej 70%).")
    else:
        st.write("Ci zawodnicy nie mieszczą się w jednej szufladce – "
                 "ich główny styl nie przekracza 70% przynależności.")
        st.dataframe(na_granicy[["styl_glowny", "styl_glowny_proc",
                                 "styl_drugi", "styl_drugi_proc", "mecze"]])

    st.subheader("Podział na jedną grupę (KMeans)")

    if os.path.exists("wyniki/wykres_style.png"):
        st.image("wyniki/wykres_style.png")

    kol1, kol2 = st.columns(2)

    with kol1:
        st.write("**Liczebność grup**")
        st.dataframe(style["styl"].value_counts())

    with kol2:
        st.write("**Średnie w grupach**")
        srednie = style.groupby("styl")[["asy", "bledy", "pierwszy_serwis", "asy_stracone"]].mean()
        st.dataframe(srednie.round(2))

    # filtr: pokaz zawodnikow z wybranej grupy
    wybrany_styl = st.selectbox("Pokaż zawodników ze stylu", sorted(style["styl"].unique()))
    grupa = style[style["styl"] == wybrany_styl].sort_values("mecze", ascending=False)
    st.dataframe(grupa[["asy", "bledy", "pierwszy_serwis", "asy_stracone", "mecze"]].round(2))

    if os.path.exists("wyniki/wykres_lokiec.png"):
        st.write("**Jak wybraliśmy liczbę grup – metoda łokcia**")
        st.image("wyniki/wykres_lokiec.png")


# ZAKŁADKA 5: RANKINGI
with z5:
    kol1, kol2 = st.columns(2)

    with kol1:
        st.subheader("Elo ogólne – TOP 20")
        st.dataframe(ranking.head(20).round(0))

    with kol2:
        st.subheader("Elo – " + nawierzchnia + " – TOP 20")
        wybrane = ranking_naw[ranking_naw.index.str.endswith("_" + nawierzchnia)]
        wybrane = wybrane.sort_values("elo", ascending=False).head(20)
        wybrane.index = wybrane.index.str.replace("_" + nawierzchnia, "", regex=False)
        st.dataframe(wybrane.round(0))

    st.caption("Ranking ATP liczy punkty za turnieje z ostatnich 12 miesięcy. "
               "Elo aktualizuje się po każdym meczu i uwzględnia siłę przeciwnika, "
               "więc szybciej pokazuje zmianę formy.")


st.caption("Dane: Jeff Sackmann (tennis_atp), licencja CC BY-NC-SA 4.0")
