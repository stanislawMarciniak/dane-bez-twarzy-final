# #     logger.info("=== KONIEC ===")
import re
import time
import logging
from rapidfuzz.distance import Levenshtein
import morfeusz2
from functools import lru_cache

# ================= LOGOWANIE =================
logger = logging.getLogger("detailed_labels")
logger.setLevel(logging.INFO)  # Zmieniono z DEBUG na INFO
# Wyłączam logowanie w tym przykładzie, aby nie tworzyć pliku, 
# ale pozostawiam konfigurację, zgodnie z oryginalnym kodem.
# logging.disable(logging.CRITICAL) 

fh = logging.FileHandler("detailed_labels.log", encoding="utf8")
fh.setLevel(logging.INFO)  # Zmieniono z DEBUG
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)  # Zmieniono z INFO
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info("=== START DETAILED LABELS ===")

# ================= KONFIGURACJA =================
FILE_ORIGINAL = "data/orig.txt"
FILE_ANONYMIZED = "data/anonimized.txt"
FILE_OUTPUT = "outputs/wyniki.txt"

KEEP_LABELS = {"name", "surname", "city", "sex", "relative", "job-title", "sexual-orientation"}
TOKEN_RE = re.compile(r'(\[[a-zA-Z0-9-]+\])|(\w+)|(\s+)|([^\w\s\[\]]+)')

PRZYPADKI = {
    "nom": "mianownik",
    "gen": "dopełniacz",
    "dat": "celownik",
    "acc": "biernik",
    "inst": "narzędnik",
    "loc": "miejscownik",
    "voc": "wołacz"
}

# Cache dla Morfeusza - kluczowa optymalizacja!
morfeusz = morfeusz2.Morfeusz()
MORFEUSZ_CACHE = {}
logger.info("Uruchomiono Morfeusz2 z cache'owaniem")

# ================= FUNKCJE =================
def tokenize_keep_delimiters(text):
    """Zoptymalizowana tokenizacja"""
    tokens = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
    return tokens

def extract_przypadek(tag_string):
    """Wyciąga pierwszy pasujący przypadek z tag_string - zoptymalizowana"""
    if not tag_string:
        return None
    
    # Szybkie sprawdzenie bez split jeśli to możliwe
    for case_key, case_name in PRZYPADKI.items():
        if case_key in tag_string:
            return case_name
    return None

def extract_rodzaj_from_tagparts(tag_parts):
    """Zamapuj symbole z tagów na 'męski' / 'żeński' - zoptymalizowana"""
    if not tag_parts:
        return None
    
    # Jeden przebieg z wczesnym returnem
    for t in tag_parts:
        if 'f' in t:
            return "woman"
        if any(m in t for m in ('m1', 'm2', 'm3', 'm')):
            return "man"
    return None

def analyse_with_cache(word):
    """Cache dla wywołań Morfeusz2 - KLUCZOWA OPTYMALIZACJA"""
    if word not in MORFEUSZ_CACHE:
        try:
            MORFEUSZ_CACHE[word] = morfeusz.analyse(word)
        except Exception as e:
            logger.error(f"Błąd morfeusz dla '{word}': {e}")
            MORFEUSZ_CACHE[word] = []
    return MORFEUSZ_CACHE[word]

def analizuj_slowo_city(tokens):
    """Zoptymalizowana analiza miasta"""
    kand_full = " ".join(tokens)
    
    for tok in tokens:
        analizy = analyse_with_cache(tok)
        
        for item in analizy:
            base = item[2][0] if isinstance(item[2], tuple) else item[2]
            tags = item[2][2] if isinstance(item[2], tuple) else ""
            dodatkowe = item[3] if len(item) > 3 else []
            
            if "nazwa_geograficzna" in dodatkowe or "subst" in tags:
                przypadek = extract_przypadek(tags)
                if przypadek:
                    return kand_full, przypadek
    
    return kand_full, None

def analizuj_slowo_sex(tokens):
    """Zoptymalizowana analiza płci"""
    kand_full = " ".join(tokens)
    analizy = analyse_with_cache(kand_full)
    
    for item in analizy:
        # Base i tags są w trzecim elemencie krotki
        if len(item) >= 3 and isinstance(item[2], tuple):
            base = item[2][0]
            tags = item[2][2]
        else:
            base = None
            tags = None
        
        if tags and "subst" in tags:
            przypadek = extract_przypadek(tags)
            if przypadek:
                return kand_full, przypadek
    
    return kand_full, None

def analizuj_slowo(slowo, label):
    """Zoptymalizowana główna analiza słowa"""
    analizy = analyse_with_cache(slowo)
    
    if not analizy:
        return None, None, None
    
    for item in analizy:
        # Wyodrębnienie base i tags z item[2]
        base = None
        tags = None
        if len(item) >= 3 and isinstance(item[2], tuple):
            base = item[2][0]
            tags = item[2][2]
        elif len(item) >= 3 and isinstance(item[2], str):
             # Prostszy przypadek, jeśli Morfeusz zwraca tylko string
            base = item[2]
            tags = ""
        
        dodatkowe = item[3] if len(item) > 3 else []
        
        if not base:
            continue
        
        tag_parts = []
        if tags:
            for part in tags.split(":"):
                tag_parts.extend(part.split("."))
        
        przypadek = extract_przypadek(tags) if tags else None
        rodzaj = extract_rodzaj_from_tagparts(tag_parts)
        
        # Heurystyka dla rodzaju - jeśli nie wykryto rodzaju, a słowo kończy się na 'a', zakładamy 'żeński'
        if not rodzaj and isinstance(base, str) and base.endswith("a"):
            rodzaj = "woman"
        
        # Akceptujemy, jeśli tag zawiera 'subst' (rzeczownik), 'adj' (przymiotnik) lub są dodatkowe informacje
        accept = (tags and ("subst" in tags or "adj" in tags)) or bool(dodatkowe)
        
        if accept and przypadek:
            return base, rodzaj, przypadek
    
    return None, None, None

def process_text_tokenized(original, anonymized, allowed_labels):
    """Zoptymalizowane przetwarzanie tekstu, korzystające z Levenshtein.opcodes"""
    
    # Tokenizacja raz na początku
    orig_tokens = tokenize_keep_delimiters(original)
    anon_tokens = tokenize_keep_delimiters(anonymized)
    
    # Obliczenie operacji edycyjnych, co pozwala zsynchronizować tokeny
    ops = Levenshtein.opcodes(anon_tokens, orig_tokens)
    output = []
    
    # Pre-kompiluj regex dla czyszczenia tokenów
    cleanup_re = re.compile(r'[.,;:(){}\[\]\n]+') # Dodałem \n do czyszczenia
    
    for tag, i1, i2, j1, j2 in ops:
        anon_chunk = anon_tokens[i1:i2]
        
        if tag == "equal":
            output.extend(anon_chunk)
        elif tag == "replace":
            # Iterujemy po tokenach w sekcji replace z pliku anonimizowanego
            for idx, token in enumerate(anon_chunk):
                # Sprawdzamy, czy token jest etykietą [LABEL]
                if not (token.startswith('[') and token.endswith(']')):
                    output.append(token)
                    continue
                
                label_name = token[1:-1]
                # Pomijamy etykiety, których nie chcemy analizować
                if label_name not in allowed_labels:
                    output.append(token)
                    continue
                
                # Ustalanie indeksu w oryginalnym tekście
                orig_idx = j1 + idx
                if orig_idx >= len(orig_tokens):
                    output.append(token)
                    continue
                
                # --- Logika analizy gramatycznej ---
                
                if label_name == "city":
                    # Heurystyka dla miast - szukamy ciągu tokenów zaczynających się z dużej litery
                    city_tokens = []
                    # Zaczynamy od potencjalnego pierwszego słowa miasta w tekście oryginalnym
                    for t_idx in range(orig_idx, len(orig_tokens)):
                        # Czyścimy słowo z interpunkcji
                        t = cleanup_re.sub('', orig_tokens[t_idx])
                        # Bierzemy tylko słowa z dużej litery (wielowyrazowe nazwy miast)
                        if t and t[0].isupper():
                            city_tokens.append(t)
                        else:
                            break # Przerywamy, jeśli trafimy na słowo niebędące częścią nazwy
                    
                    _, przypadek = analizuj_slowo_city(city_tokens)
                    tag_new = f"[{label_name}][{przypadek}]" if przypadek else f"[{label_name}]"
                    output.append(tag_new)
                
                elif label_name == "sex":
                    # Analiza płci
                    kand = cleanup_re.sub('', orig_tokens[orig_idx])
                    _, przypadek = analizuj_slowo_sex([kand])
                    tag_new = f"[{label_name}][{przypadek}]" if przypadek else f"[{label_name}]"
                    output.append(tag_new)
                
                else:
                    # Ogólna analiza dla pozostałych etykiet
                    kand = cleanup_re.sub('', orig_tokens[orig_idx])
                    base, rodzaj, przypadek = analizuj_slowo(kand, label_name)
                    
                    if base and rodzaj and przypadek:
                        tag_new = f"[{label_name}][{rodzaj}][{przypadek}]"
                    else:
                        # W przypadku niepowodzenia analizy, zostawiamy oryginalną etykietę [LABEL]
                        tag_new = token 
                    output.append(tag_new)
        
        elif tag == "delete":
            # Tokeny usunięte z pliku oryginalnego (widoczne w anonimizowanym, ale nie w oryginalnym)
            output.extend(anon_chunk)
        # insert - operacje wstawienia (insert) z operacji Levenshtein,
        # czyli tokeny występujące tylko w tekście oryginalnym (orig_tokens[j1:j2]),
        # są ignorowane, ponieważ pracujemy na tokenach z tekstu anonimizowanego.

    return "".join(output)

def read_file(filepath):
    """Odczytuje zawartość pliku, obsługując błąd FileNotFoundError."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error("Brak pliku: %s", filepath)
        # print(f"Brak pliku: {filepath}") # Można włączyć dla szybkiego debugowania
        return None

def save_file(filepath, content):
    """Zapisuje zawartość do pliku."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Zapisano wynik do: %s", filepath)

start = time.perf_counter()
orig = read_file(FILE_ORIGINAL)
anon = read_file(FILE_ANONYMIZED)

if orig and anon:
    print("\n================= PRZETWARZANIE =================")
    
    result = process_text_tokenized(orig, anon, KEEP_LABELS)
    save_file(FILE_OUTPUT, result)
    
    end = time.perf_counter()
    cache_size = len(MORFEUSZ_CACHE)
    
    print(f"✔ Przetworzono w {end - start:.3f} sekund")
    print(f"✔ Cache Morfeusz: {cache_size} unikalnych słów")
    print(f"✔ Wynik zapisano do: {FILE_OUTPUT}")
else:
    print("\n================= BŁĄD PLIKÓW =================")
    print("Nie można przetworzyć tekstu: brak jednego lub obu plików źródłowych.")
    
logger.info("=== KONIEC ===")