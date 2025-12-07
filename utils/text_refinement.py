import re
import time
from collections import defaultdict

# --- KONFIGURACJA ---
FILE_ORIGINAL = "data/orig.txt"
FILE_ANONYMIZED = "data/anonimized.txt"
FILE_OUTPUT = "output_final1.txt"

KEEP_LABELS = {
    "name", "surname", "age", "date-of-birth", "date", "sex", "religion",
    "political-view", "ethnicity", "sexual-orientation", "health", "relative",
    "city", "address",
    "phone", "document-number",
    "company", "school-name", "job-title",
    "bank-account", "credit-card-number",
    "credit-card-number",
    "username", "secret"
}

# Pattern taga: przechwytuje wszystko bez spacji w nawiasach kwadratowych
TAG_PATTERN = re.compile(r'\[([^\s\]]+)\]')

WORD_RE = re.compile(r'\w+', flags=re.UNICODE)

# --- funkcje pomocnicze ---

def find_tags_spans(text):
    """Zwraca listę (start, end, tag_text) dla wszystkich tagów w tekście."""
    return [(m.start(), m.end(), m.group(1)) for m in TAG_PATTERN.finditer(text)]


def process_text_tokenized(original, anonymized, allowed_labels):
    # 1. Szybka tokenizacja
    orig_tokens = tokenize_keep_delimiters(original)
    anon_tokens = tokenize_keep_delimiters(anonymized)

    # 2. RapidFuzz – generowanie opcodów (SUPER SZYBKIE)
    ops = Levenshtein.opcodes(anon_tokens, orig_tokens)

    output = []

    for tag, i1, i2, j1, j2 in ops:

        # Pobranie fragmentów
        anon_chunk = anon_tokens[i1:i2]
        orig_chunk = orig_tokens[j1:j2]

        if tag == "equal":
            output.extend(anon_chunk)

        elif tag == "replace":
            # Zastępuj zawsze jeśli jest [phone] lub forbidden
            has_phone = any("[phone]" in t for t in anon_chunk)
            contains_forbidden = any(
                t.startswith("[") and t.endswith("]") and t[1:-1] not in allowed_labels
                for t in anon_chunk
            )
            if has_phone or contains_forbidden:
                output.extend(orig_chunk)
            else:
                output.extend(anon_chunk)


        elif tag == "delete":
            # To jest w anon, ale nie ma w oryginale → przepisujemy
            output.extend(anon_chunk)

        elif tag == "insert":
            # Ignorujemy — tak samo jak w twojej wersji
            pass

    return "".join(output)


def group_adjacent_tags(tag_spans, text):
    """
    Grupuje kolejne tagi.
    """
    if not tag_spans:
        return []

    separators_re = re.compile(r'^[\s,;:\-—()\[\]{}"\'`«»…]*$')

    groups = []
    current = [tag_spans[0]]
    for prev, cur in zip(tag_spans, tag_spans[1:]):
        prev_end = prev[1]
        cur_start = cur[0]
        between = text[prev_end:cur_start]
        if separators_re.match(between):
            current.append(cur)
        else:
            groups.append((current[0][0], current[-1][1], list(current)))
            current = [cur]
    groups.append((current[0][0], current[-1][1], list(current)))
    return groups


def find_replacement_between_anchors(orig_line, word_before, word_after):
    """
    Szuka w oryginalnej linii fragmentu pomiędzy anchorami.
    """
    flags = re.DOTALL
    if word_before and word_after:
        pat = r'\b' + re.escape(word_before) + r'\b(.*?)\b' + re.escape(word_after) + r'\b'
        m = re.search(pat, orig_line, flags)
        if m:
            return m.group(1)
        pat2 = r'\b' + re.escape(word_before) + r'\b(.*?)(?=' + re.escape(word_after) + r')'
        m2 = re.search(pat2, orig_line, flags)
        if m2:
            return m2.group(1)
    elif word_before:
        pat = r'\b' + re.escape(word_before) + r'\b(.*)$'
        m = re.search(pat, orig_line, flags)
        if m:
            return m.group(1)
    elif word_after:
        pat = r'^(.*?)(?=\b' + re.escape(word_after) + r'\b)'
        m = re.search(pat, orig_line, flags)
        if m:
            return m.group(1)
    return None


def find_anchor_positions_in_anon(non_tag_tokens, group_start, group_end):
    """
    Znajduje pozycje anchorów w tekście zanonimizowanym.
    """
    word_before = None
    word_after = None
    anon_segment_start = 0
    anon_segment_end = None

    for w, s, e in reversed(non_tag_tokens):
        if e <= group_start:
            word_before = w
            anon_segment_start = e
            break

    for w, s, e in non_tag_tokens:
        if s >= group_end:
            word_after = w
            anon_segment_end = s
            break

    return anon_segment_start, anon_segment_end, word_before, word_after


def fallback_extract_by_type(orig_line, tag_list):
    return None


def process_line(anon_line, orig_line, keep_labels, line_number, examples_registry):
    """
    Przetwarza linię i zbiera przykłady dla tagów zachowanych (KEEP_LABELS).
    Argument examples_registry to słownik: {tag_name: [list_of_examples]}
    """
    result = anon_line
    tag_spans = find_tags_spans(result)
    if not tag_spans:
        return result

    groups = group_adjacent_tags(tag_spans, result)

    # Procesujemy od końca
    for gstart, gend, tags in reversed(groups):
        
        # --- LOGIKA ZBIERANIA PRZYKŁADÓW DLA KEEP_LABELS ---
        # Jeżeli wszystkie tagi w grupie są w KEEP_LABELS
        if all(t[2] in keep_labels for t in tags):
            # Sprawdź, czy potrzebujemy przykładów dla któregokolwiek z tych tagów (czy mamy mniej niż 5)
            need_example = any(len(examples_registry[t[2]]) < 5 for t in tags)
            
            if need_example:
                # Używamy tej samej logiki co do zamiany, żeby wyciągnąć oryginał
                non_tag_tokens = non_tag_tokens_with_spans(result, tag_spans)
                _, _, word_before, word_after = find_anchor_positions_in_anon(non_tag_tokens, gstart, gend)
                
                original_content = find_replacement_between_anchors(orig_line, word_before, word_after)
                
                if original_content and original_content.strip():
                    cleaned_content = original_content.strip()
                    # Zapisujemy przykład dla każdego taga w tej grupie
                    for t in tags:
                        tag_name = t[2]
                        if len(examples_registry[tag_name]) < 5:
                            # Unikamy duplikatów w przykładach
                            if cleaned_content not in examples_registry[tag_name]:
                                examples_registry[tag_name].append(cleaned_content)
            
            # Kontynuujemy (nie podmieniamy tagów z KEEP_LABELS)
            continue

        # --- Dalsza część bez zmian (podmiana tagów spoza KEEP_LABELS) ---
        non_tag_tokens = non_tag_tokens_with_spans(result, tag_spans)
        anon_seg_start, anon_seg_end, word_before, word_after = find_anchor_positions_in_anon(non_tag_tokens, gstart, gend)

        if anon_seg_end is None:
            anon_seg_end_idx = len(result)
        else:
            anon_seg_end_idx = anon_seg_end

        replacement_between = find_replacement_between_anchors(orig_line, word_before, word_after)

        if replacement_between is not None and replacement_between.strip() != "":
            result = result[:anon_seg_start] + replacement_between + result[anon_seg_end_idx:]
            tag_spans = find_tags_spans(result)
            continue

        replacement_fallback = fallback_extract_by_type(orig_line, tags)
        if replacement_fallback is not None and str(replacement_fallback).strip() != "":
            result = result[:anon_seg_start] + replacement_fallback + result[anon_seg_end_idx:]
            tag_spans = find_tags_spans(result)
            continue

        tag_texts = ','.join([t[2] for t in tags])
        print(f"NIE ZNALEZIONO ORYGINAŁU DLA: [{tag_texts}] (przed: '{word_before}', po: '{word_after}') w linii: {line_number}")
        return None

    return result


def process_files(file_original, file_anonymized, file_output, keep_labels):
    try:
        with open(file_anonymized, 'r', encoding='utf-8') as f:
            anon_lines = f.readlines()
    except FileNotFoundError:
        print(f"Brak pliku: {file_anonymized}")
        return

    try:
        with open(file_original, 'r', encoding='utf-8') as f:
            orig_lines = f.readlines()
    except FileNotFoundError:
        print(f"Brak pliku: {file_original}")
        return

    if len(anon_lines) != len(orig_lines):
        print(f"Ostrzeżenie: Różna liczba linii - anonimizowany: {len(anon_lines)}, oryginalny: {len(orig_lines)}")
        min_lines = min(len(anon_lines), len(orig_lines))
        anon_lines = anon_lines[:min_lines]
        orig_lines = orig_lines[:min_lines]

    output_lines = []
    removed_count = 0
    
    # Słownik na przykłady: klucz to nazwa taga, wartość to lista znalezionych oryginałów
    examples_registry = defaultdict(list)

    for i, (a, o) in enumerate(zip(anon_lines, orig_lines)):
        anon_line = a.rstrip('\n')
        orig_line = o.rstrip('\n')
        
        # Przekazujemy examples_registry do funkcji
        processed = process_line(anon_line, orig_line, keep_labels, line_number=i+1, examples_registry=examples_registry)
        
        if processed is None:
            removed_count += 1
        else:
            output_lines.append(processed + '\n')

        if (i + 1) % 1000 == 0:
            print(f"Przetworzono {i + 1} linii... (usunięto {removed_count})")

    with open(file_output, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"\nZapisano wynik: {file_output}")
    print(f"Liczba oryginalnych linii: {len(anon_lines)}, zapisanych linii: {len(output_lines)}, usuniętych linii: {removed_count}")
    
    # --- WYPISANIE PRZYKŁADÓW ---
    print("\n" + "="*50)
    print("PRZYKŁADY ORYGINALNYCH WARTOŚCI DLA ZACHOWANYCH TAGÓW:")
    print("="*50)
    
    # Sortujemy tagi alfabetycznie
    sorted_tags = sorted(examples_registry.keys())
    
    for tag in sorted_tags:
        examples = examples_registry[tag]
        if examples:
            print(f"\nTAG: [{tag}]")
            for idx, ex in enumerate(examples, 1):
                print(f"  {idx}. {ex}")
    print("="*50)


if __name__ == '__main__':
    start = time.perf_counter()
    print("Wczytywanie i przetwarzanie plików...")
    process_files(FILE_ORIGINAL, FILE_ANONYMIZED, FILE_OUTPUT, KEEP_LABELS)
    end = time.perf_counter()
    print(f"Czas wykonania: {end - start:.4f} sekundy")