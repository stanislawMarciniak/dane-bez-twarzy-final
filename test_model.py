from transformers import pipeline
import os

# Ścieżka do folderu z rozpakowanym modelem
MODEL_PATH = "./models" 
# MODEL_PATH = "./final_ner_model" 

def extend_entity_boundaries(text, start, end, entity_text, tag_type):
    """
    Zaawansowane dociąganie granic w zależności od typu tagu.
    """
    text_len = len(text)
    extended_end = end

    # --- 1. SPECJALNE TRAKTOWANIE DLA CIĄGŁYCH ZNAKÓW (DOC_NUM, EMAIL, PESEL) ---
    # Te tagi nie powinny mieć spacji w środku, ale mogą mieć litery, cyfry i znaki specjalne.
    # Reguła: "Bierz wszystko aż do spacji".
    if tag_type in ['DOC_NUM', 'PESEL', 'EMAIL', 'IBAN', 'CREDIT_CARD']:
        while extended_end < text_len:
            char = text[extended_end]
            # Jeśli trafimy na spację (lub enter/tab) -> KONIEC
            if char.isspace():
                break
            
            # Opcjonalnie: Jeśli trafimy na kropkę lub przecinek NA KOŃCU zdania, 
            # to technicznie nie jest część numeru, ale dla bezpieczeństwa anonimizacji
            # lepiej zakryć o jeden znak za dużo niż za mało.
            # Jeśli chcesz być precyzyjny i nie zjadać kropki na końcu zdania:
            if char in ['.', ',', '!', '?'] and (extended_end + 1 == text_len or text[extended_end+1].isspace()):
                break
            
            extended_end += 1
        return start, extended_end

    # --- 2. SPECJALNE TRAKTOWANIE DLA TELEFONÓW (PHONE) ---
    # Telefony są wredne, bo mają spacje w środku (np. 500 123 456).
    # Reguła: "Skacz po spacjach/myślnikach, jeśli dalej są cyfry".
    if tag_type == 'PHONE' or any(char.isdigit() for char in entity_text):
        while extended_end < text_len:
            current_char = text[extended_end]
            
            # A. Cyfra -> bierzemy
            if current_char.isdigit():
                extended_end += 1
            # B. Separator (spacja, -), ale TYLKO jeśli zaraz za nim jest cyfra
            elif current_char in [' ', '-'] and (extended_end + 1 < text_len) and text[extended_end+1].isdigit():
                extended_end += 1
            # C. Inne znaki -> koniec
            else:
                break
        return start, extended_end
            
    # --- 3. DOMYŚLNA LOGIKA DLA TEKSTU (NAME, CITY itp.) ---
    # Bierzemy wszystko do spacji (np. doklejamy końcówki fleksyjne lub interpunkcję)
    while extended_end < text_len:
        char = text[extended_end]
        if char.isspace():
            break
        extended_end += 1

    return start, extended_end

def anonymize_text(text, nlp_model):
    # 1. Pobranie surowych wyników
    results = nlp_model(text)
    
    if not results:
        return text

    # 2. Sortowanie po indeksie startowym
    results = sorted(results, key=lambda x: x['start'])

    output = ""
    current_idx = 0
    last_processed_end = -1

    for entity in results:
        original_start = entity['start']
        original_end = entity['end']
        tag = entity['entity_group']
        word_fragment = entity['word']

        # Pomijamy nakładające się encje
        if original_start < last_processed_end:
            continue

        # 3. Rozszerzanie granic
        new_start, new_end = extend_entity_boundaries(text, original_start, original_end, word_fragment, tag)

        # Przepisujemy tekst PRZED encją
        output += text[current_idx:new_start]

        # 4. Wstawiamy Tag
        formatted_tag = f"[{tag.lower()}]"
        output += formatted_tag

        # Aktualizujemy indeksy
        current_idx = new_end
        last_processed_end = new_end

    # 5. Doklejamy resztę tekstu
    output += text[current_idx:]
    
    return output

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD: Nie znaleziono folderu {MODEL_PATH}.")
        return

    print("Ładowanie modelu...")
    
    try:
        nlp = pipeline(
            "token-classification", 
            model=MODEL_PATH, 
            aggregation_strategy="simple" 
        )
    except Exception as e:
        print(f"Wystąpił błąd podczas ładowania modelu: {e}")
        return

    print("\n✅ Model gotowy! Wpisz zdanie. (wpisz 'q' by wyjść)")
    print("-" * 50)

    while True:
        try:
            text = input("\nTekst: ")
        except EOFError:
            break
        
        if text.lower() in ['q', 'exit', 'quit']:
            break
            
        if not text.strip():
            continue

        result_text = anonymize_text(text, nlp)
        
        print("\n--- WYNIK ---")
        print(result_text)
        print("-" * 20)

if __name__ == "__main__":
    main()