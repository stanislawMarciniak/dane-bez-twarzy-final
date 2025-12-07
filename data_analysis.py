import re
from collections import defaultdict, Counter
from pathlib import Path

# --- KONFIGURACJA ---
FILE_OUTPUT = "data/model_data.txt"

# Regex taki sam jak w skrypcie przetwarzającym
TAG_PATTERN = re.compile(r'\[([a-zA-Z0-9-]+)\]')

def analyze_file(file_path):
    path = Path(file_path)
    
    if not path.exists():
        print(f"Błąd: Plik {file_path} nie istnieje.")
        return

    print(f"Rozpoczynam analizę pliku: {file_path} ...\n")

    total_lines = 0
    lines_without_tags = 0
    total_tags_count = 0
    
    # Słownik do zliczania wystąpień każdego taga
    tag_counter = Counter()
    
    # Słownik do przechowywania przykładów: { 'tag': [lista_zdan] }
    tag_examples = defaultdict(list)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            
            # Znajdź wszystkie tagi w linii
            found_tags = TAG_PATTERN.findall(line)
            
            if not found_tags:
                lines_without_tags += 1
            else:
                total_tags_count += len(found_tags)
                
                # Aktualizuj statystyki dla każdego znalezionego taga
                for tag in found_tags:
                    tag_counter[tag] += 1
                    
                    # Dodaj przykład jeśli mamy mniej niż 5 i ta linia jeszcze nie jest w przykładach dla tego taga
                    if len(tag_examples[tag]) < 5:
                        if line not in tag_examples[tag]:
                            tag_examples[tag].append(line)

    # --- RAPORT ---
    
    print("=== PODSUMOWANIE OGÓLNE ===")
    print(f"Liczba przetworzonych linii: {total_lines}")
    print(f"Liczba wszystkich wykrytych tagów: {total_tags_count}")
    print(f"Liczba linii BEZ żadnego taga: {lines_without_tags}")
    if total_lines > 0:
        percent_clean = (lines_without_tags / total_lines) * 100
        print(f"Procent linii 'czystych' (bez tagów): {percent_clean:.2f}%")
    print("-" * 40)
    print("\n=== SZCZEGÓŁY WG RODZAJU TAGA ===")
    
    # Sortujemy tagi od najczęstszych
    sorted_tags = tag_counter.most_common()
    
    for tag, count in sorted_tags:
        print(f"\nTAG: [{tag}] (Wystąpień: {count})")


if __name__ == "__main__":
    analyze_file(FILE_OUTPUT)