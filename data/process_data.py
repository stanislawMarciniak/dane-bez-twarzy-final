import json
import spacy
from spacy.training import offsets_to_biluo_tags

# Pobierz model spacy (jeśli nie masz, odkomentuj linię download)
try:
    nlp = spacy.load("pl_core_news_sm")
except OSError:
    from spacy.cli import download
    download("pl_core_news_sm")
    nlp = spacy.load("pl_core_news_sm")

def filter_overlaps(entities):
    """
    Usuwa nakładające się encje, preferując te dłuższe.
    Input: lista krotek (start, end, label)
    Output: przefiltrowana lista krotek posortowana po start_index
    """
    # 1. Sortujemy po długości malejąco (najdłuższe mają pierwszeństwo)
    #    W przypadku równej długości, bierzemy te, które zaczynają się wcześniej
    sorted_by_len = sorted(entities, key=lambda x: (x[1] - x[0], -x[0]), reverse=True)
    
    kept_entities = []
    
    # Utwórz zbiór zajętych indeksów znaków
    occupied_indices = set()
    
    for start, end, label in sorted_by_len:
        # Sprawdź, czy jakikolwiek indeks w tym zakresie jest już zajęty
        # range(start, end) generuje indeksy znaków od start do end-1
        current_indices = set(range(start, end))
        
        # Jeśli część wspólna jest pusta (brak konfliktu)
        if not occupied_indices.intersection(current_indices):
            kept_entities.append((start, end, label))
            occupied_indices.update(current_indices)
            
    # Spacy wymaga, aby encje były posortowane rosnąco po indeksie startowym
    return sorted(kept_entities, key=lambda x: x[0])

def convert_to_ner_format(input_file, output_file):
    training_data = []
    
    TAG_MAP = {
        "[name]": "NAME", "[surname]": "SURNAME", "[age]": "AGE", 
        "[sex]": "SEX", "[address]": "ADDRESS", "[phone]": "PHONE", 
        "[document-number]": "DOC_NUM", "[health]": "HEALTH", 
        "[relative]": "RELATIVE", "[city]": "CITY", "[company]": "COMPANY",
        "[date]": "DATE", "[credit-card-number]": "CREDIT_CARD",
        "[job-title]": "JOB", "[school-name]": "SCHOOL",
        "[bank-account]": "IBAN", "[ethnicity]": "ETHNICITY",
        "[political-view]": "POLITICS", "[religion]": "RELIGION",
        "[sexual-orientation]": "ORIENTATION", "[username]": "USER",
        "[secret]": "SECRET"
    }

    print(f"Przetwarzanie {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get('result', '')
            replacements = data.get('replacements', {})
            
            if not text: 
                continue

            # Lista wszystkich znalezionych encji (potencjalnie z konfliktami)
            raw_entities = []
            
            # Iterujemy po replacements
            for tag, value in replacements.items():
                if not value:
                    continue
                    
                val_str = str(value)
                clean_tag = TAG_MAP.get(tag, "UNKNOWN")
                
                # Znajdź WSZYSTKIE wystąpienia danej wartości w tekście, nie tylko pierwsze
                # Używamy pętli while, aby znaleźć duplikaty (np. jeśli imię pada 2 razy)
                start_search = 0
                while True:
                    start_index = text.find(val_str, start_search)
                    if start_index == -1:
                        break
                    
                    end_index = start_index + len(val_str)
                    raw_entities.append((start_index, end_index, clean_tag))
                    
                    # Przesuwamy szukanie dalej
                    start_search = end_index

            # >>> KLUCZOWA ZMIANA: FILTROWANIE KONFLIKTÓW <<<
            final_entities = filter_overlaps(raw_entities)

            # Spacy tokenizuje tekst i dopasowuje tagi BIO
            try:
                doc = nlp(text)
                tags = offsets_to_biluo_tags(doc, final_entities)
            except Exception as e:
                print(f"Pominięto linię {line_num} z powodu błędu spacy: {e}")
                continue
            
            # Konwersja BILUO na standardowe BIO
            bio_tags = []
            for tag in tags:
                tag = tag.replace("L-", "I-").replace("U-", "B-")
                bio_tags.append(tag)

            tokens = [token.text for token in doc]
            
            # Zapisujemy tylko jeśli tagowanie się udało
            if "-" not in bio_tags: 
                training_data.append({"tokens": tokens, "ner_tags": bio_tags})

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Sukces! Przygotowano {len(training_data)} przykładów w {output_file}")

if __name__ == "__main__":
    convert_to_ner_format("data_final.jsonl", "data_ner_ready.jsonl")