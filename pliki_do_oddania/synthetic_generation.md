# Moduł Generacji Danych Syntetycznych – Overfitters

Nasz moduł generacji danych syntetycznych wykorzystuje analizator morfologiczny **Morfeusz** w celu precyzyjnej odmiany słów w języku polskim zgodnie z przypadkiem gramatycznym i płcią.

---

## Kluczowe funkcje

### 1. Odmiana fleksyjna
Moduł przyjmuje tagi określające przypadek (np. `[miejscownik]`) oraz płeć (`[man]` / `[woman]`).  
Dzięki temu możliwe jest poprawne odmienianie imion, nazwisk, miast, zawodów oraz nazw instytucji i firm.

**Przykłady:**  
- `Radom → Radomiu [miejscownik]`
- `Anna → Anny [dopełniacz]`  
- `inżynier → inżyniera [biernik]`

### 2. Losowość i anonimowość
- Każdy token losuje unikalną wartość z puli danych.  
- Pula danych stopniowo się zuboża, eliminując powtórzenia.  
- Losowanie gwarantuje całkowite oderwanie danych syntetycznych od oryginalnego kontekstu (zgodność z RODO).

### 3. Obsługa różnych typów danych
Moduł generuje m.in.:
- Imiona i nazwiska (mężczyzn i kobiet)  
- Miasta i miejscowości  
- Nazwy firm i szkół  
- Zawody i stanowiska  
- Dane kontaktowe: e-mail, telefon, adres  
- Identyfikatory: PESEL, numery kont bankowych, numery dokumentów, numery kart kredytowych  
- Dane zdrowotne i relacje rodzinne  

---

## Przykłady zastosowania

| Tekst źródłowy                                                          | Tokeny                                                                                                                  | Tekst syntetyczny                                                      |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Jestem specjalistą ds. kadr w TechNet S.A.                              | Jestem [job-title][man][narzędnik] w [company][miejscownik]                                                             | Jestem kierownikiem budowy w EkoBud Sp. z o.o.                         |
| Chcę jechać do Poznania razem z tatą mojej mamy.                        | Chcę jechać do [city][dopełniacz] razem z [relative][man][narzędnik] mojej [relative][woman][dopełniacz]                | Chcę jechać do Wrocławia razem z bratem mojej babci                    |
| Rozmawialiśmy o Adamie Kowalskim. Ma 35 lat i ostatnio był w Warszawie. | Rozmawialiśmy o [name][man][miejscownik] [surname][man][miejscownik]. Ma [age] lat i ostatnio był w [city][miejscownik] | Rozmawialiśmy o Tomaszu Zalewskim. Ma 50 lat i ostatnio był w Krakowie |
| Spotkałem się z Kasią i jej mamą w Gdańsku.                             | Spotkałem się z [name][woman][narzędnik] i jej [relative][woman][narzędnik]                                             | Spotkałem się z Hanną i jej siostrą                                    |
| Jestem mężczyzną i ateistą.                                             | Jestem [sex][man][narzędnik] i [religion][man][narzędnik]                                                               | Jestem mężczyzną i ewangelikiem                                        |

Przykłądowy log:

--- Oryginalna Zanonimizowana Treść ---
Rozmawialiśmy o [name][man][miejscownik] [surname][man][miejscownik]. Ma [age] lat i ostatnio był w [city][miejscownik]

--- Treść Syntetyczna ---
Rozmawialiśmy o Piotrze Szymańskim. Ma 35 lat i ostatnio był w Poznaniu

---

## Zalety modułu

- **Poprawność fleksyjna i gramatyczna:** każde słowo jest odmieniane zgodnie z przypadkiem i płcią.  
- **Zachowanie spójności językowej:** nawet przy losowych danych tekst pozostaje naturalny i czytelny.  
- **Wszechstronność:** obsługa różnorodnych typów danych osobowych, zawodowych i instytucjonalnych.  
- **Anonimowość:** brak powiązania danych syntetycznych z oryginalnym kontekstem, pełna zgodność z RODO.  
- **Łatwość integracji:** moduł działa na prostych tokenach `[name]`, `[city]`, `[job-title]`, `[company]`, `[school-name]`.

---

## Podsumowanie
Moduł stanowi skuteczne narzędzie do generowania danych syntetycznych dla systemów testowych, raportów, analiz lub szkoleń AI, które wymagają poprawnych, naturalnie brzmiących danych w języku polskim.  

Dzięki połączeniu fleksji, losowości i bogatych pul danych możliwe jest generowanie dużych zbiorów danych całkowicie bezpiecznych i zgodnych z przepisami o ochronie danych osobowych.
