#!/usr/bin/env python3
"""
Pipeline - Zunifikowany proces anonimizacji i generacji syntetycznej.
WERSJA ZOPTYMALIZOWANA: Batch Processing + GPU Support.

Kolejność przetwarzania:
1. Model ML - wykrywa encje NER
2. Regex Layer - łapie email, PESEL, telefony itp.
3. Zapis do pliki_do_oddania/output_Overfitters.txt
4. Detailed Labels - dodaje info o płci i przypadku
5. Synthetic Generator - generuje dane syntetyczne
6. Zapis do pliki_do_oddania/synthetic_generation_Overfitters.txt
"""

import sys
import os
import time
import torch  # Do wykrywania GPU
from typing import Optional, Dict
from dataclasses import dataclass, field

# === KONFIGURACJA ===
MODEL_PATH = "./models"
OUTPUT_DIR = "./pliki_do_oddania"
OUTPUT_ANONYMIZED = "output_Overfitters.txt"
OUTPUT_SYNTHETIC = "synthetic_generation_Overfitters.txt"

# Parametry wydajności
BATCH_SIZE = 32  # Przetwarzanie 32 linii naraz (zwiększ jeśli masz mocne GPU)
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU


@dataclass
class TimingResult:
    """Przechowuje czasy wykonania poszczególnych etapów."""
    model_load_time: float = 0.0  # Nie wliczany do total
    ml_layer_time: float = 0.0
    regex_layer_time: float = 0.0
    detailed_labels_time: float = 0.0
    synthetic_generation_time: float = 0.0
    file_io_time: float = 0.0
    
    # Czasy kumulatywne (bez ładowania modelu)
    time_to_anonymized: float = 0.0  # Do output_Overfitters.txt
    time_to_synthetic: float = 0.0   # Do synthetic_generation_Overfitters.txt
    total_time: float = 0.0
    
    # Statystyki per sample
    num_samples: int = 0
    avg_time_per_sample: float = 0.0
    avg_ml_per_sample: float = 0.0
    avg_regex_per_sample: float = 0.0
    avg_detailed_per_sample: float = 0.0
    avg_synthetic_per_sample: float = 0.0
    
    def calculate_averages(self):
        """Oblicza średnie czasy per sample."""
        if self.num_samples > 0:
            self.avg_time_per_sample = self.total_time / self.num_samples
            self.avg_ml_per_sample = self.ml_layer_time / self.num_samples
            self.avg_regex_per_sample = self.regex_layer_time / self.num_samples
            self.avg_detailed_per_sample = self.detailed_labels_time / self.num_samples
            self.avg_synthetic_per_sample = self.synthetic_generation_time / self.num_samples
    
    def __str__(self) -> str:
        self.calculate_averages()
        
        # Formatowanie ms dla małych wartości
        def fmt_time(t):
            if t < 0.001:
                return f"{t*1000000:.1f} µs"
            elif t < 1:
                return f"{t*1000:.2f} ms"
            else:
                return f"{t:.3f} s"
        
        return f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                         ⏱️  POMIAR CZASU                              ║
║                   (bez ładowania modelu/bibliotek)                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║ Warstwa ML (NER):              {self.ml_layer_time:>10.3f} s    │ avg: {fmt_time(self.avg_ml_per_sample):>12}  ║
║ Warstwa Regex:                 {self.regex_layer_time:>10.3f} s    │ avg: {fmt_time(self.avg_regex_per_sample):>12}  ║
║ Detailed Labels:               {self.detailed_labels_time:>10.3f} s    │ avg: {fmt_time(self.avg_detailed_per_sample):>12}  ║
║ Generacja syntetyczna:         {self.synthetic_generation_time:>10.3f} s    │ avg: {fmt_time(self.avg_synthetic_per_sample):>12}  ║
║ Zapis plików (I/O):            {self.file_io_time:>10.3f} s    │                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 📄 Czas do outputOverfitters:  {self.time_to_anonymized:>10.3f} s                            ║
║ 📄 Czas do synthetic_gen:      {self.time_to_synthetic:>10.3f} s                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 📊 Liczba próbek (linii):      {self.num_samples:>10}                                 ║
║ 📊 Średni czas per sample:     {fmt_time(self.avg_time_per_sample):>12}                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 🏁 CAŁKOWITY CZAS:             {self.total_time:>10.3f} s                            ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# === IMPORTY MODUŁÓW PROJEKTU ===

# 1. Model ML
from transformers import pipeline as hf_pipeline

# 2. Regex Layer
from .regex_layer import RegexLayer, EntityType

# 3. Detailed Labels
from .detailed_labels import process_text_tokenized, KEEP_LABELS

# 4. Synthetic Generator
from .synthetic_generator import generate_synthetic_output


# === FUNKCJE ANONIMIZACJI ML ===

def extend_entity_boundaries(text, start, end, entity_text, tag_type):
    """
    Zaawansowane dociąganie granic w zależności od typu tagu.
    WAŻNE: NIE rozszerzamy na znaki interpunkcyjne jak ), ., ,, !, ?, ;, :
    Obsługuje rozszerzanie WSTECZ dla PHONE i NAME/SURNAME.
    """
    text_len = len(text)
    extended_start = start
    extended_end = end
    
    # Znaki interpunkcyjne które ZAWSZE kończą encję
    PUNCTUATION_STOP = set('.,;:!?)]}"\'\n\r\t')

    # Usuwamy prefixy B-/I- dla pewności porównania w ifach
    clean_type = tag_type.upper().replace("B-", "").replace("I-", "")

    # SPECJALNE TRAKTOWANIE DLA CIĄGŁYCH ZNAKÓW (DOC_NUM, EMAIL, PESEL)
    if clean_type in ['DOC_NUM', 'PESEL', 'EMAIL', 'IBAN', 'CREDIT_CARD', 'USERNAME']:
        while extended_end < text_len:
            char = text[extended_end]
            if char.isspace() or char in PUNCTUATION_STOP:
                break
            extended_end += 1
        return extended_start, extended_end

    # SPECJALNE TRAKTOWANIE DLA TELEFONÓW (PHONE)
    if clean_type == 'PHONE':
        # Rozszerzanie W PRZÓD - do kolejnych cyfr
        while extended_end < text_len:
            current_char = text[extended_end]
            if current_char in PUNCTUATION_STOP:
                break
            if current_char.isdigit():
                extended_end += 1
            elif current_char in [' ', '-'] and (extended_end + 1 < text_len) and text[extended_end+1].isdigit():
                extended_end += 1
            else:
                break
        
        # Rozszerzanie WSTECZ - do + (włącznie) lub litery (wyłącznie)
        while extended_start > 0:
            prev_char = text[extended_start - 1]
            if prev_char == '+':
                # Włączamy + i kończymy
                extended_start -= 1
                break
            elif prev_char.isalpha():
                # Litera - STOP (nie włączamy)
                break
            elif prev_char.isdigit() or prev_char in [' ', '-']:
                # Cyfra lub separator - włączamy
                extended_start -= 1
            else:
                # Inny znak - STOP
                break
        
        return extended_start, extended_end

    # SPECJALNE TRAKTOWANIE DLA NAME/SURNAME - rozszerzanie wstecz do spacji
    if clean_type in ['NAME', 'SURNAME']:
        # Rozszerzanie W PRZÓD - do końca słowa
        while extended_end < text_len:
            char = text[extended_end]
            if char.isspace() or char in PUNCTUATION_STOP:
                break
            if char.isalnum() or char in ['-', "'", '"']:
                extended_end += 1
            else:
                break
        
        # Rozszerzanie WSTECZ - do spacji (wyłącznie)
        while extended_start > 0:
            prev_char = text[extended_start - 1]
            if prev_char.isspace() or prev_char in PUNCTUATION_STOP:
                # Spacja lub interpunkcja - STOP (nie włączamy)
                break
            elif prev_char.isalnum() or prev_char in ['-', "'", '"']:
                # Litera/cyfra - włączamy
                extended_start -= 1
            else:
                # Inny znak - STOP
                break
        
        return extended_start, extended_end
            
    # DOMYŚLNA LOGIKA DLA TEKSTU (CITY, AGE itp.)
    # Rozszerzaj tylko na litery/cyfry, STOP na interpunkcji i białych znakach
    while extended_end < text_len:
        char = text[extended_end]
        if char.isspace() or char in PUNCTUATION_STOP:
            break
        if char.isalnum() or char in ['-', "'", '"']:
            extended_end += 1
        else:
            break

    return extended_start, extended_end


def normalize_tag(tag: str) -> str:
    """
    Normalizuje tag z modelu NER do formatu [lowercase].
    """
    clean_tag = tag.upper().replace("B-", "").replace("I-", "")
    
    tag_mapping = {
        'NAME': 'name', 'SURNAME': 'surname', 'AGE': 'age', 'SEX': 'sex',
        'CITY': 'city', 'ADDRESS': 'address', 'PHONE': 'phone', 'EMAIL': 'email',
        'PESEL': 'pesel', 'DATE': 'date', 'RELATIVE': 'relative',
        'JOB': 'job-title', 'COMPANY': 'company', 'SCHOOL': 'school-name',
        'HEALTH': 'health', 'RELIGION': 'religion', 'POLITICS': 'political-view',
        'ETHNICITY': 'ethnicity', 'ORIENTATION': 'sexual-orientation',
        'IBAN': 'bank-account', 'CREDIT_CARD': 'credit-card-number',
        'DOC_NUM': 'document-number', 'USER': 'username', 'SECRET': 'secret',
        # UNKNOWN jest pomijany - nie tagujemy
    }
    return tag_mapping.get(clean_tag, clean_tag.lower())


def apply_ner_to_line(line: str, results: list) -> str:
    """
    Nakłada wyniki NER na linię tekstu.
    Ta funkcja NIE wywołuje modelu, tylko przetwarza wyniki.
    """
    if not line.strip() or not results:
        return line

    results = sorted(results, key=lambda x: x['start'])

    output = ""
    current_idx = 0
    last_processed_end = -1

    for entity in results:
        original_start = entity['start']
        original_end = entity['end']
        raw_tag = entity['entity_group']
        word_fragment = entity['word']
        
        # Usuwamy prefix B-/I- dla porównania
        clean_tag = raw_tag.upper().replace("B-", "").replace("I-", "")
        
        # POMIJAMY encje UNKNOWN - zostawiamy oryginalny tekst
        if clean_tag == 'UNKNOWN':
            continue

        if original_start < last_processed_end:
            continue

        # 1. Obliczamy granice (używając SUROWEGO tagu)
        new_start, new_end = extend_entity_boundaries(
            line, original_start, original_end, word_fragment, raw_tag
        )

        # 2. Normalizujemy tag do wyświetlenia
        display_tag = normalize_tag(raw_tag)

        # Przepisujemy tekst PRZED encją
        output += line[current_idx:new_start]

        # Wstawiamy Tag
        output += f"[{display_tag}]"

        current_idx = new_end
        last_processed_end = new_end

    # Doklejamy resztę tekstu
    output += line[current_idx:]
    return output


def ml_anonymize_text(text: str, nlp_model, show_progress: bool = True) -> str:
    """
    ZOPTYMALIZOWANA Anonimizacja: Batch Processing.
    """
    lines = text.split('\n')
    
    # Wyciągamy tylko niepuste linie, żeby nie marnować GPU
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    non_empty_lines = [lines[i] for i in non_empty_indices]
    
    if not non_empty_lines:
        return text

    # Uruchamiamy model w trybie wsadowym (Batch)
    # To jest kluczowe przyspieszenie - model dostaje listę, a nie pojedyncze stringi
    if show_progress:
        print(f"🚀 Przetwarzanie ML w batchach (Batch size: {BATCH_SIZE}, Device: {DEVICE})...")
    
    batch_results = nlp_model(non_empty_lines, batch_size=BATCH_SIZE)
    
    # Rekonstrukcja tekstu
    processed_lines = lines.copy()
    
    for idx, line_results in zip(non_empty_indices, batch_results):
        processed_lines[idx] = apply_ner_to_line(lines[idx], line_results)
    
    return '\n'.join(processed_lines)


# === REGEX LAYER ===

def regex_anonymize_text(text: str, regex_layer: RegexLayer) -> str:
    """
    Anonimizacja tekstu przez warstwę regex.
    Łapie: email, PESEL, telefony, numery kont, adresy.
    """
    entities = regex_layer.detect(text)
    
    if not entities:
        return text
    
    # Sortuj od końca, żeby nie psuć indeksów przy zamianie
    entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)
    
    result = text
    for entity in entities_sorted:
        tag = f"[{entity.entity_type.value}]"
        result = result[:entity.start] + tag + result[entity.end:]
    
    return result


# === GŁÓWNY PIPELINE ===

class AnonymizationPipeline:
    def __init__(self, model_path: str = MODEL_PATH, verbose: bool = True, output_dir: str = OUTPUT_DIR):
        self.verbose = verbose
        self.model_path = model_path
        self.output_dir = output_dir
        self.nlp_model = None
        self.regex_layer = None
        self.timing = TimingResult()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def load_models(self):
        t_start = time.perf_counter()
        self._log("📦 Ładowanie modelu ML...")
        
        if not os.path.exists(self.model_path):
             self._log(f"⚠️ Nie znaleziono {self.model_path}. Używam domyślnego HerBERTa.")
             self.model_path = "allegro/herbert-base-cased"
        
        # Kluczowe: parametr device dla GPU
        self.nlp_model = hf_pipeline(
            "token-classification",
            model=self.model_path,
            aggregation_strategy="simple",
            device=DEVICE 
        )
        
        self.timing.model_load_time = time.perf_counter() - t_start
        self._log(f"✅ Model ML załadowany ({self.timing.model_load_time:.3f}s) [Device: {DEVICE}]")
        
        self._log("📦 Inicjalizacja warstwy Regex...")
        self.regex_layer = RegexLayer()
        self._log("✅ Warstwa Regex gotowa.")
    
    def process(self, original_text: str, output_anonymized: str = OUTPUT_ANONYMIZED, output_synthetic: str = OUTPUT_SYNTHETIC) -> dict:
        if self.nlp_model is None or self.regex_layer is None:
            self.load_models()
        
        self.timing = TimingResult() # Reset
        pipeline_start = time.perf_counter()
        
        # Liczenie linii
        lines = original_text.strip().split('\n')
        self.timing.num_samples = len([l for l in lines if l.strip()])
        
        results = {
            'original': original_text,
            'after_ml': None,
            'after_regex': None,
            'after_detailed_labels': None,
            'synthetic': None,
            'timing': None
        }
        
        # === ETAP 1: Model ML (BATCHED) ===
        self._log("\n🔹 ETAP 1: Anonimizacja ML (Batch)")
        t_start = time.perf_counter()
        after_ml = ml_anonymize_text(original_text, self.nlp_model)
        self.timing.ml_layer_time = time.perf_counter() - t_start
        results['after_ml'] = after_ml
        
        # === ETAP 2: Regex Layer ===
        self._log("🔹 ETAP 2: Regex")
        t_start = time.perf_counter()
        after_regex = regex_anonymize_text(after_ml, self.regex_layer)
        self.timing.regex_layer_time = time.perf_counter() - t_start
        results['after_regex'] = after_regex
        
        # ZAPIS 1
        t_io_start = time.perf_counter()
        out_path = os.path.join(self.output_dir, output_anonymized)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(after_regex)
        t_io_anon = time.perf_counter() - t_io_start
        self.timing.time_to_anonymized = time.perf_counter() - pipeline_start
        
        # === ETAP 3: Detailed Labels ===
        self._log("🔹 ETAP 3: Detailed Labels")
        cpu_count = os.cpu_count()
        self._log(f"🖥️  Dostępne rdzenie CPU: {cpu_count}")
        t_start = time.perf_counter()
        after_detailed = process_text_tokenized(original_text, after_regex, KEEP_LABELS)
        self.timing.detailed_labels_time = time.perf_counter() - t_start
        results['after_detailed_labels'] = after_detailed
        
        # === ETAP 4: Synthetic ===
        self._log("🔹 ETAP 4: Synthetic Gen")
        t_start = time.perf_counter()
        synthetic = generate_synthetic_output(after_detailed)
        self.timing.synthetic_generation_time = time.perf_counter() - t_start
        results['synthetic'] = synthetic
        
        # ZAPIS 2
        t_io_start_2 = time.perf_counter()
        out_path_syn = os.path.join(self.output_dir, output_synthetic)
        with open(out_path_syn, 'w', encoding='utf-8') as f:
            f.write(synthetic)
        t_io_synth = time.perf_counter() - t_io_start_2
        
        self.timing.file_io_time = t_io_anon + t_io_synth
        self.timing.time_to_synthetic = time.perf_counter() - pipeline_start
        self.timing.total_time = self.timing.time_to_synthetic
        
        self._log(f"✅ Zapisano: {out_path}")
        self._log(f"✅ Zapisano: {out_path_syn}")
        self._log(str(self.timing))
        
        results['timing'] = self.timing
        return results

    def process_file(self, input_file: str, output_anonymized=OUTPUT_ANONYMIZED, output_synthetic=OUTPUT_SYNTHETIC):
        self._log(f"📂 Wczytuję plik: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.process(text, output_anonymized, output_synthetic)


# === BLOK URUCHAMIAJĄCY ===

def main():
    print("\n" + "="*60)
    print("   🔐 PIPELINE ANONIMIZACJI (OPTIMIZED)")
    print("="*60)
    
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        pipeline = AnonymizationPipeline()
        pipeline.process_file(input_file)
    else:
        print("\nTryb interaktywny (wpisz tekst, Ctrl+D lub 'q' by wyjść).")
        pipeline = AnonymizationPipeline()
        pipeline.load_models()
        
        while True:
            try:
                text = input("\n> ")
                if text.strip().lower() in ['q', 'exit']: break
                if not text.strip(): continue
                
                res = pipeline.process(text)
                print("\n--- WYNIK ---")
                print(res['synthetic'])
                
            except EOFError:
                break

if __name__ == "__main__":
    main()