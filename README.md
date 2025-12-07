# Dane bez Twarzy - Overfitters Pipeline

**System anonimizacji danych osobowych w tekstach polskich**

Biblioteka Python do automatycznej detekcji i anonimizacji danych wrażliwych w tekstach w języku polskim. Zaprojektowana dla projektu PLLuM (Polish Large Language Model).

## Cele projektu

- **Bezpieczeństwo danych** - wykrywanie i anonimizacja danych osobowych zgodnie z RODO
- **Zachowanie struktury** - podmiana encji na tokeny zachowujące sens i gramatykę
- **Wsparcie dla polskiego** - pełna obsługa fleksji i kontekstu językowego
- **Wydajność** - skalowalne rozwiązanie do przetwarzania dużych zbiorów danych
- **Generacja syntetyczna** - tworzenie realistycznych danych zastępczych

## Architektura Pipeline

System wykorzystuje wielowarstwową architekturę:

```
┌─────────────────────────────────────────────────────────────┐
│                        WEJŚCIE                              │
│              "Jan Kowalski, PESEL 90010112345"              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               ETAP 1: MODEL ML (NER)                        │
│                 Wytrenowany model NER                       │
│  • Imiona i nazwiska                                        │
│  • Miasta, adresy                                           │
│  • Wiek, płeć, relacje rodzinne                             │
│  • Firmy, stanowiska                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               ETAP 2: WARSTWA REGEX                         │
│                    "Szybkie Sito"                           │
│  • PESEL (z walidacją sumy kontrolnej)                      │
│  • Email, telefon, IBAN                                     │
│  • Daty (różne formaty)                                     │
│  • Numery dokumentów                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    📄 outputOverfitters.txt
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           ETAP 3: DETAILED LABELS (Morfologia)              │
│               Analiza gramatyczna (Morfeusz2)               │
│  • Przypadek (mianownik, dopełniacz, biernik...)            │
│  • Rodzaj (męski, żeński)                                   │
│  • Format: [name][man][mianownik]                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           ETAP 4: GENERATOR SYNTETYCZNY                     │
│            Tworzenie realistycznych danych                  │
│  • Imiona i nazwiska z puli                                 │
│  • Odmiana przez przypadki                                  │
│  • Generowanie PESEL, email, telefon                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
               📄 synthetic_generation_Overfitters.txt
```

## Instalacja

### Wymagania

- Python 3.8+
- pip

### Instalacja

```bash
# Klonowanie repozytorium
git clone git@github.com:stanislawMarciniak/dane-bez-twarzy.git
cd dane-bez-twarzy

# Stworzenie wirtualnego środowiska
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub: venv\Scripts\activate  # Windows

# Instalacja zależności
pip install -r requirements.txt
```

### Wymagane zależności

```
transformers>=4.20.0    # Model ML (NER)
torch>=1.10.0           # Backend dla transformers
morfeusz2>=1.99.12      # Analiza morfologiczna polskiego
rapidfuzz>=3.6.1        # Algorytm Levenshtein
```

## Użycie

### Interfejs wiersza poleceń (CLI)

```bash
# Przetwarzanie pliku
python pipeline.py data/orig.txt

# Wyniki zapisywane automatycznie do:
# - pliki_do_oddania/outputOverfitters.txt
# - pliki_do_oddania/synthetic_generation_Overfitters.txt
```

### Jako moduł Python

```bash
python -m overfitters_pipeline.pipeline data/orig.txt
```

### API Python

```python
from overfitters_pipeline import AnonymizationPipeline

# Inicjalizacja pipeline
pipeline = AnonymizationPipeline(
    model_path="./models",      # Ścieżka do modelu NER
    output_dir="./pliki_do_oddania",
    verbose=True
)

# Przetwarzanie tekstu
results = pipeline.process("""
Nazywam się Jan Kowalski, mój PESEL to 90010112345.
Mieszkam w Warszawie przy ul. Długiej 5.
Mój email to jan.kowalski@gmail.com, telefon: +48 123 456 789.
""")

# Wyniki
print(results['after_ml'])              # Po modelu ML
print(results['after_regex'])           # Po warstwie regex (outputOverfitters)
print(results['after_detailed_labels']) # Z etykietami morfologicznymi
print(results['synthetic'])             # Dane syntetyczne

# Statystyki czasowe
print(results['timing'])
```

### Przetwarzanie pliku

```python
from overfitters_pipeline import AnonymizationPipeline

pipeline = AnonymizationPipeline()
results = pipeline.process_file("data/orig.txt")

# Pliki wyjściowe:
# - pliki_do_oddania/outputOverfitters.txt
# - pliki_do_oddania/synthetic_generation_Overfitters.txt
```

## Struktura projektu

```
dane-bez-twarzy/
├── overfitters_pipeline/           # Główny pakiet pipeline
│   ├── __init__.py
│   ├── pipeline.py                 # Główny pipeline z mierzeniem czasu
│   ├── regex_layer.py              # Warstwa regex
│   ├── detailed_labels.py          # Etykiety morfologiczne (Morfeusz2)
│   ├── synthetic_generator.py      # Generator danych syntetycznych
│   ├── synthetic_data_pool.py      # Pule danych syntetycznych
│   └── morfeusz_inflector.py       # Odmiana przez przypadki
├── models/                         # Wytrenowany model NER
├── data/                           # Dane wejściowe
├── pliki_do_oddania/               # Wyniki (outputy)
│   ├── outputOverfitters.txt
│   └── synthetic_generation_Overfitters.txt
├── pipeline.py                     # Wrapper CLI
├── requirements.txt
└── README.md
```

## Obsługiwane kategorie danych

### 1. Dane identyfikacyjne osobowe

| Token                  | Opis                 | Przykład            |
| ---------------------- | -------------------- | ------------------- |
| `[name]`               | Imiona               | Jan, Anna           |
| `[surname]`            | Nazwiska             | Kowalski, Nowak     |
| `[age]`                | Wiek                 | 25 lat              |
| `[date-of-birth]`      | Data urodzenia       | 15.03.1990          |
| `[date]`               | Inne daty            | przyjęto 23.09.2023 |
| `[sex]`                | Płeć                 | mężczyzna, kobieta  |
| `[religion]`           | Wyznanie             | katolik             |
| `[political-view]`     | Poglądy polityczne   | konserwatysta       |
| `[ethnicity]`          | Pochodzenie etniczne | Ukrainiec           |
| `[sexual-orientation]` | Orientacja seksualna | heteroseksualny     |
| `[health]`             | Dane zdrowotne       | cukrzyca typu 2     |
| `[relative]`           | Relacje rodzinne     | mój brat Piotr      |

### 2. Dane kontaktowe i lokalizacyjne

| Token       | Opis                        | Przykład                     |
| ----------- | --------------------------- | ---------------------------- |
| `[city]`    | Miasto (lokalizacja ogólna) | Jadę do Krakowa              |
| `[address]` | Pełny adres                 | ul. Długa 5, 00-001 Warszawa |
| `[email]`   | Adres email                 | jan@gmail.com                |
| `[phone]`   | Numer telefonu              | +48 123 456 789              |

### 3. Identyfikatory dokumentów

| Token               | Opis              | Przykład    |
| ------------------- | ----------------- | ----------- |
| `[pesel]`           | Numer PESEL       | 90010112345 |
| `[document-number]` | Numery dokumentów | ABC123456   |

### 4. Dane zawodowe i edukacyjne

| Token           | Opis         | Przykład               |
| --------------- | ------------ | ---------------------- |
| `[company]`     | Nazwa firmy  | TechPol Sp. z o.o.     |
| `[school-name]` | Nazwa szkoły | Uniwersytet Warszawski |
| `[job-title]`   | Stanowisko   | programista            |

### 5. Informacje finansowe

| Token                  | Opis        | Przykład            |
| ---------------------- | ----------- | ------------------- |
| `[bank-account]`       | Numer konta | PL12 1234 ...       |
| `[credit-card-number]` | Numer karty | 4111 1111 1111 1111 |

### 6. Identyfikatory cyfrowe

| Token        | Opis                    | Przykład        |
| ------------ | ----------------------- | --------------- |
| `[username]` | Login/nazwa użytkownika | @janek123       |
| `[secret]`   | Hasła, klucze API       | hasło: tajne123 |

## Format etykiet szczegółowych

Po przetworzeniu przez `detailed_labels.py`, etykiety zawierają informacje morfologiczne:

```
[name][man][mianownik]    # Imię męskie w mianowniku
[city][dopełniacz]        # Miasto w dopełniaczu
[relative][woman][biernik] # Relacja żeńska w bierniku
```

## Pomiar czasu

Pipeline automatycznie mierzy i wyświetla czasy wykonania:

```
╔═══════════════════════════════════════════════════════════════════════╗
║                         ⏱️  POMIAR CZASU                              ║
║                   (bez ładowania modelu/bibliotek)                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║ Warstwa ML (NER):                   5.219 s    │ avg:      1.69 ms  ║
║ Warstwa Regex:                     34.738 s    │ avg:     11.22 ms  ║
║ Detailed Labels:                    1.922 s    │ avg:     620.9 µs  ║
║ Generacja syntetyczna:              0.046 s    │ avg:      15.0 µs  ║
║ Zapis plików (I/O):                 0.079 s    │                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 📄 Czas do outputOverfitters:      40.014 s                          ║
║ 📄 Czas do synthetic_gen:          42.012 s                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 📊 Liczba próbek (linii):            3096                            ║
║ 📊 Średni czas per sample:         13.57 ms                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 🏁 CAŁKOWITY CZAS:                 42.012 s                          ║
╚═══════════════════════════════════════════════════════════════════════╝
```

**Uwaga:** Czas ładowania modelu nie jest wliczany do całkowitego czasu.

## Przykład użycia

### Wejście
```
O kurde, muszę się wyżalić, bo zaraz eksploduję. Jestem Piotr, Noras, 79 lat, 
kobieta. Mieszkam w Dębicy przy ul. Wrzozowej 10, a jakby coś, to łap kontakt: 
795 324 908 albo wezyktola@example.net.
```

### Po ML + Regex (outputOverfitters.txt)
```
O kurde, muszę się wyżalić, bo zaraz eksploduję. Jestem [name] Noras, [age] lat, 
[sex] Mieszkam w [city] przy [address] a jakby coś, to łap kontakt: 
[phone] albo [email].
```

### Po Detailed Labels
```
O kurde, muszę się wyżalić, bo zaraz eksploduję. Jestem [name] Noras, [age] lat, 
[sex] Mieszkam w [city][dopełniacz] przy [address] a jakby coś, to łap kontakt: 
[phone] albo [email].
```

### Dane syntetyczne (synthetic_generation_Overfitters.txt)
```
O kurde, muszę się wyżalić, bo zaraz eksploduję. Jestem Tomasz Noras, 47 lat, 
kobieta Mieszkam w Gdyni przy pl. Akacjowa 63, 31-846 Kraków a jakby coś, 
to łap kontakt: +48 649 878 486 albo kowalski16@gmail.com.
```

## Zespół

**Overfitters** - projekt realizowany w ramach PLLuM (Polish Large Language Model)
