"""
Detailed Labels - Dodawanie etykiet morfologicznych (płeć, przypadek).
Zrównoleglone przetwarzanie z wykorzystaniem wszystkich rdzeni CPU.
"""

import re
import os
import logging
from rapidfuzz.distance import Levenshtein
import morfeusz2
from functools import lru_cache
from multiprocessing import Pool, cpu_count, current_process

# ================= LOGOWANIE =================
logger = logging.getLogger("detailed_labels")
logger.setLevel(logging.WARNING)  # Zmniejszamy verbose

# ================= KONFIGURACJA =================
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

# Regex do czyszczenia tokenów
CLEANUP_RE = re.compile(r'[.,;:(){}\[\]\n]+')

# ================= FUNKCJE POMOCNICZE =================

def get_num_workers():
    """Zwraca liczbę rdzeni CPU do użycia."""
    return cpu_count() or 1


def tokenize_keep_delimiters(text):
    """Zoptymalizowana tokenizacja"""
    tokens = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
    return tokens


def extract_przypadek(tag_string):
    """Wyciąga pierwszy pasujący przypadek z tag_string"""
    if not tag_string:
        return None
    
    for case_key, case_name in PRZYPADKI.items():
        if case_key in tag_string:
            return case_name
    return None


def extract_rodzaj_from_tagparts(tag_parts):
    """Zamapuj symbole z tagów na 'man' / 'woman'"""
    if not tag_parts:
        return None
    
    for t in tag_parts:
        if 'f' in t:
            return "woman"
        if any(m in t for m in ('m1', 'm2', 'm3', 'm')):
            return "man"
    return None


# ================= KLASA PROCESORA (dla multiprocessing) =================

class LineProcessor:
    """
    Procesor pojedynczej linii z własną instancją Morfeusz2.
    Każdy proces workera tworzy własną instancję.
    """
    
    _morfeusz = None
    _cache = None
    
    @classmethod
    def init_worker(cls):
        """Inicjalizacja workera - tworzy instancję Morfeusz2 dla procesu."""
        cls._morfeusz = morfeusz2.Morfeusz()
        cls._cache = {}
    
    @classmethod
    def analyse_with_cache(cls, word):
        """Cache dla wywołań Morfeusz2"""
        if cls._morfeusz is None:
            cls.init_worker()
        
        if word not in cls._cache:
            try:
                cls._cache[word] = cls._morfeusz.analyse(word)
            except Exception:
                cls._cache[word] = []
        return cls._cache[word]
    
    @classmethod
    def analizuj_slowo_city(cls, tokens):
        """Analiza miasta"""
        kand_full = " ".join(tokens)
        
        for tok in tokens:
            analizy = cls.analyse_with_cache(tok)
            
            for item in analizy:
                base = item[2][0] if isinstance(item[2], tuple) else item[2]
                tags = item[2][2] if isinstance(item[2], tuple) else ""
                dodatkowe = item[3] if len(item) > 3 else []
                
                if "nazwa_geograficzna" in dodatkowe or "subst" in tags:
                    przypadek = extract_przypadek(tags)
                    if przypadek:
                        return kand_full, przypadek
        
        return kand_full, None
    
    @classmethod
    def analizuj_slowo_sex(cls, tokens):
        """Analiza płci"""
        kand_full = " ".join(tokens)
        analizy = cls.analyse_with_cache(kand_full)
        
        for item in analizy:
            if len(item) >= 3 and isinstance(item[2], tuple):
                tags = item[2][2]
            else:
                tags = None
            
            if tags and "subst" in tags:
                przypadek = extract_przypadek(tags)
                if przypadek:
                    return kand_full, przypadek
        
        return kand_full, None
    
    @classmethod
    def analizuj_slowo(cls, slowo, label):
        """Główna analiza słowa"""
        analizy = cls.analyse_with_cache(slowo)
        
        if not analizy:
            return None, None, None
        
        for item in analizy:
            base = None
            tags = None
            if len(item) >= 3 and isinstance(item[2], tuple):
                base = item[2][0]
                tags = item[2][2]
            elif len(item) >= 3 and isinstance(item[2], str):
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
            
            if not rodzaj and isinstance(base, str) and base.endswith("a"):
                rodzaj = "woman"
            
            accept = (tags and ("subst" in tags or "adj" in tags)) or bool(dodatkowe)
            
            if accept and przypadek:
                return base, rodzaj, przypadek
        
        return None, None, None


def _process_single_line(args):
    """
    Przetwarza pojedynczą linię (dla multiprocessing).
    Args: (original_line, anonymized_line, allowed_labels)
    """
    original_line, anonymized_line, allowed_labels = args
    
    if not original_line.strip() or not anonymized_line.strip():
        return anonymized_line
    
    orig_tokens = tokenize_keep_delimiters(original_line)
    anon_tokens = tokenize_keep_delimiters(anonymized_line)
    
    ops = Levenshtein.opcodes(anon_tokens, orig_tokens)
    output = []
    
    for tag, i1, i2, j1, j2 in ops:
        anon_chunk = anon_tokens[i1:i2]
        
        if tag == "equal":
            output.extend(anon_chunk)
        elif tag == "replace":
            for idx, token in enumerate(anon_chunk):
                if not (token.startswith('[') and token.endswith(']')):
                    output.append(token)
                    continue
                
                label_name = token[1:-1]
                if label_name not in allowed_labels:
                    output.append(token)
                    continue
                
                orig_idx = j1 + idx
                if orig_idx >= len(orig_tokens):
                    output.append(token)
                    continue
                
                if label_name == "city":
                    city_tokens = []
                    for t_idx in range(orig_idx, len(orig_tokens)):
                        t = CLEANUP_RE.sub('', orig_tokens[t_idx])
                        if t and t[0].isupper():
                            city_tokens.append(t)
                        else:
                            break
                    
                    _, przypadek = LineProcessor.analizuj_slowo_city(city_tokens)
                    tag_new = f"[{label_name}][{przypadek}]" if przypadek else f"[{label_name}]"
                    output.append(tag_new)
                
                elif label_name == "sex":
                    kand = CLEANUP_RE.sub('', orig_tokens[orig_idx])
                    _, przypadek = LineProcessor.analizuj_slowo_sex([kand])
                    tag_new = f"[{label_name}][{przypadek}]" if przypadek else f"[{label_name}]"
                    output.append(tag_new)
                
                else:
                    kand = CLEANUP_RE.sub('', orig_tokens[orig_idx])
                    base, rodzaj, przypadek = LineProcessor.analizuj_slowo(kand, label_name)
                    
                    if base and rodzaj and przypadek:
                        tag_new = f"[{label_name}][{rodzaj}][{przypadek}]"
                    else:
                        tag_new = token
                    output.append(tag_new)
        
        elif tag == "delete":
            output.extend(anon_chunk)
    
    return "".join(output)


def _init_pool_worker():
    """Inicjalizacja workera w puli procesów."""
    LineProcessor.init_worker()


# ================= GŁÓWNA FUNKCJA PRZETWARZANIA =================

def process_text_tokenized(original, anonymized, allowed_labels, num_workers=None):
    """
    Zrównoleglone przetwarzanie tekstu z użyciem wszystkich rdzeni CPU.
    
    Args:
        original: Oryginalny tekst
        anonymized: Zanonimizowany tekst
        allowed_labels: Zbiór etykiet do przetworzenia
        num_workers: Liczba workerów (domyślnie = liczba rdzeni CPU)
    
    Returns:
        Przetworzony tekst z etykietami morfologicznymi
    """
    # Określ liczbę workerów
    if num_workers is None:
        num_workers = get_num_workers()
    
    # Podziel na linie
    orig_lines = original.split('\n')
    anon_lines = anonymized.split('\n')
    
    # Wyrównaj długość list
    max_lines = max(len(orig_lines), len(anon_lines))
    while len(orig_lines) < max_lines:
        orig_lines.append('')
    while len(anon_lines) < max_lines:
        anon_lines.append('')
    
    num_lines = len(anon_lines)
    
    # Dla małej liczby linii użyj przetwarzania sekwencyjnego
    if num_lines < 10 or num_workers == 1:
        LineProcessor.init_worker()
        results = []
        for orig_line, anon_line in zip(orig_lines, anon_lines):
            result = _process_single_line((orig_line, anon_line, allowed_labels))
            results.append(result)
        return '\n'.join(results)
    
    # Przygotuj argumenty dla workerów
    args_list = [
        (orig_lines[i], anon_lines[i], allowed_labels)
        for i in range(num_lines)
    ]
    
    # Przetwarzanie równoległe
    with Pool(processes=num_workers, initializer=_init_pool_worker) as pool:
        results = pool.map(_process_single_line, args_list)
    
    return '\n'.join(results)


# ================= FUNKCJA DLA PIPELINE =================

def process_text_tokenized_with_info(original, anonymized, allowed_labels):
    """
    Wrapper z informacją o liczbie używanych rdzeni.
    Zwraca: (wynik, liczba_rdzeni)
    """
    num_workers = get_num_workers()
    result = process_text_tokenized(original, anonymized, allowed_labels, num_workers)
    return result, num_workers


# ================= TESTY =================

if __name__ == "__main__":
    import time
    
    print(f"\n🖥️  Dostępne rdzenie CPU: {get_num_workers()}")
    
    # Test
    FILE_ORIGINAL = "data/orig.txt"
    FILE_ANONYMIZED = "data/anonimized.txt"
    FILE_OUTPUT = "outputs/wyniki.txt"
    
    def read_file(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Brak pliku: {filepath}")
            return None
    
    def save_file(filepath, content):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    start = time.perf_counter()
    orig = read_file(FILE_ORIGINAL)
    anon = read_file(FILE_ANONYMIZED)
    
    if orig and anon:
        print(f"\n================= PRZETWARZANIE ({get_num_workers()} rdzeni) =================")
        
        result = process_text_tokenized(orig, anon, KEEP_LABELS)
        save_file(FILE_OUTPUT, result)
        
        end = time.perf_counter()
        
        print(f"✔ Przetworzono w {end - start:.3f} sekund")
        print(f"✔ Wynik zapisano do: {FILE_OUTPUT}")
    else:
        print("\n================= BŁĄD PLIKÓW =================")
        print("Nie można przetworzyć tekstu: brak jednego lub obu plików źródłowych.")
