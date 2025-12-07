#!/usr/bin/env python3
"""
Pipeline - Zunifikowany proces anonimizacji i generacji syntetycznej.

Kolejność przetwarzania:
1. Model ML - wykrywa encje NER
2. Regex Layer - łapie email, PESEL, telefony itp.
3. Zapis do pliki_do_oddania/outputOverfitters.txt
4. Detailed Labels - dodaje info o płci i przypadku
5. Synthetic Generator - generuje dane syntetyczne
6. Zapis do pliki_do_oddania/synthetic_generation_Overfitters.txt

Zawiera szczegółowe mierzenie czasu dla każdej warstwy (bez ładowania modelu).
"""

import sys
import os
import time
from typing import Optional, Dict
from dataclasses import dataclass, field

# === KONFIGURACJA ===
MODEL_PATH = "./models"
OUTPUT_DIR = "./pliki_do_oddania"
OUTPUT_ANONYMIZED = "outputOverfitters.txt"
OUTPUT_SYNTHETIC = "synthetic_generation_Overfitters.txt"


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
    time_to_anonymized: float = 0.0  # Do outputOverfitters.txt
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
║ 📄 Czas do outputOverfitters:  {self.time_to_anonymized:>10.3f} s                              ║
║ 📄 Czas do synthetic_gen:      {self.time_to_synthetic:>10.3f} s                              ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 📊 Liczba próbek (linii):      {self.num_samples:>10}                                   ║
║ 📊 Średni czas per sample:     {fmt_time(self.avg_time_per_sample):>12}                               ║
╠═══════════════════════════════════════════════════════════════════════╣
║ 🏁 CAŁKOWITY CZAS:             {self.total_time:>10.3f} s                              ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# === IMPORTY MODUŁÓW PROJEKTU ===

# 1. Model ML
from transformers import pipeline as hf_pipeline

# 2. Regex Layer (z tego samego folderu)
from .regex_layer import RegexLayer, EntityType

# 3. Detailed Labels
from .detailed_labels import process_text_tokenized, KEEP_LABELS

# 4. Synthetic Generator
from .synthetic_generator import generate_synthetic_output


# === FUNKCJE ANONIMIZACJI ML ===

def extend_entity_boundaries(text, start, end, entity_text, tag_type):
    """
    Zaawansowane dociąganie granic w zależności od typu tagu.
    """
    text_len = len(text)
    extended_end = end

    # SPECJALNE TRAKTOWANIE DLA CIĄGŁYCH ZNAKÓW (DOC_NUM, EMAIL, PESEL)
    if tag_type in ['DOC_NUM', 'PESEL', 'EMAIL', 'IBAN', 'CREDIT_CARD']:
        while extended_end < text_len:
            char = text[extended_end]
            if char.isspace():
                break
            if char in ['.', ',', '!', '?'] and (extended_end + 1 == text_len or text[extended_end+1].isspace()):
                break
            extended_end += 1
        return start, extended_end

    # SPECJALNE TRAKTOWANIE DLA TELEFONÓW (PHONE)
    if tag_type == 'PHONE' or any(char.isdigit() for char in entity_text):
        while extended_end < text_len:
            current_char = text[extended_end]
            if current_char.isdigit():
                extended_end += 1
            elif current_char in [' ', '-'] and (extended_end + 1 < text_len) and text[extended_end+1].isdigit():
                extended_end += 1
            else:
                break
        return start, extended_end
            
    # DOMYŚLNA LOGIKA DLA TEKSTU (NAME, CITY itp.)
    while extended_end < text_len:
        char = text[extended_end]
        if char.isspace():
            break
        extended_end += 1

    return start, extended_end


def ml_anonymize_text(text: str, nlp_model) -> str:
    """
    Anonimizacja tekstu przez model ML.
    Zwraca tekst z tagami typu [name], [city], [age] itd.
    """
    results = nlp_model(text)
    
    if not results:
        return text

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

        # Rozszerzanie granic
        new_start, new_end = extend_entity_boundaries(
            text, original_start, original_end, word_fragment, tag
        )

        # Przepisujemy tekst PRZED encją
        output += text[current_idx:new_start]

        # Wstawiamy Tag
        formatted_tag = f"[{tag.lower()}]"
        output += formatted_tag

        # Aktualizujemy indeksy
        current_idx = new_end
        last_processed_end = new_end

    # Doklejamy resztę tekstu
    output += text[current_idx:]
    
    return output


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
    """
    Główna klasa pipeline'u anonimizacji z mierzeniem czasu.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, verbose: bool = True, output_dir: str = OUTPUT_DIR):
        self.verbose = verbose
        self.model_path = model_path
        self.output_dir = output_dir
        self.nlp_model = None
        self.regex_layer = None
        self.timing = TimingResult()
        
        # Upewnij się, że folder wyjściowy istnieje
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def load_models(self):
        """Ładuje model ML i inicjalizuje warstwę regex (nie wliczane do czasu)."""
        t_start = time.perf_counter()
        
        self._log("📦 Ładowanie modelu ML...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Nie znaleziono folderu z modelem: {self.model_path}")
        
        self.nlp_model = hf_pipeline(
            "token-classification",
            model=self.model_path,
            aggregation_strategy="simple"
        )
        
        self.timing.model_load_time = time.perf_counter() - t_start
        self._log(f"✅ Model ML załadowany ({self.timing.model_load_time:.3f}s) [NIE WLICZANE DO CZASU]")
        
        self._log("📦 Inicjalizacja warstwy Regex...")
        self.regex_layer = RegexLayer()
        self._log("✅ Warstwa Regex gotowa.")
    
    def process(
        self,
        original_text: str,
        output_anonymized: str = OUTPUT_ANONYMIZED,
        output_synthetic: str = OUTPUT_SYNTHETIC
    ) -> dict:
        """
        Przetwarza tekst przez cały pipeline z mierzeniem czasu.
        Czas ładowania modelu NIE jest wliczany do całkowitego czasu.
        """
        # Ładowanie modelu PRZED startem pomiaru czasu
        if self.nlp_model is None or self.regex_layer is None:
            self.load_models()
        
        # Reset timing i START pomiaru (PO załadowaniu modelu)
        self.timing = TimingResult()
        pipeline_start = time.perf_counter()
        
        # Liczenie linii (samples)
        lines = original_text.strip().split('\n')
        self.timing.num_samples = len([l for l in lines if l.strip()])
        
        # Pełne ścieżki wyjściowe
        output_anon_path = os.path.join(self.output_dir, output_anonymized)
        output_synth_path = os.path.join(self.output_dir, output_synthetic)
        
        results = {
            'original': original_text,
            'after_ml': None,
            'after_regex': None,
            'after_detailed_labels': None,
            'synthetic': None,
            'timing': None
        }
        
        # === ETAP 1: Model ML ===
        self._log("\n" + "="*50)
        self._log("🔹 ETAP 1: Anonimizacja przez model ML")
        self._log("="*50)
        
        t_start = time.perf_counter()
        after_ml = ml_anonymize_text(original_text, self.nlp_model)
        self.timing.ml_layer_time = time.perf_counter() - t_start
        
        results['after_ml'] = after_ml
        self._log(f"⏱️  Czas ML: {self.timing.ml_layer_time:.3f}s")
        self._log(f"Wynik ML:\n{after_ml[:300]}..." if len(after_ml) > 300 else f"Wynik ML:\n{after_ml}")
        
        # === ETAP 2: Regex Layer ===
        self._log("\n" + "="*50)
        self._log("🔹 ETAP 2: Anonimizacja przez warstwę Regex")
        self._log("="*50)
        
        t_start = time.perf_counter()
        after_regex = regex_anonymize_text(after_ml, self.regex_layer)
        self.timing.regex_layer_time = time.perf_counter() - t_start
        
        results['after_regex'] = after_regex
        self._log(f"⏱️  Czas Regex: {self.timing.regex_layer_time:.3f}s")
        self._log(f"Wynik Regex:\n{after_regex[:300]}..." if len(after_regex) > 300 else f"Wynik Regex:\n{after_regex}")
        
        # === ZAPIS DO outputOverfitters.txt ===
        t_io_start = time.perf_counter()
        self._log(f"\n💾 Zapisuję do {output_anon_path}...")
        with open(output_anon_path, 'w', encoding='utf-8') as f:
            f.write(after_regex)
        t_io_anon = time.perf_counter() - t_io_start
        
        # Czas do outputOverfitters.txt (bez ładowania modelu)
        self.timing.time_to_anonymized = time.perf_counter() - pipeline_start
        self._log(f"✅ Zapisano: {output_anon_path}")
        self._log(f"⏱️  Czas do outputOverfitters.txt: {self.timing.time_to_anonymized:.3f}s")
        
        # === ETAP 3: Detailed Labels ===
        self._log("\n" + "="*50)
        self._log("🔹 ETAP 3: Dodawanie szczegółowych etykiet (płeć, przypadek)")
        self._log("="*50)
        
        # Wyświetl liczbę rdzeni CPU
        cpu_count = os.cpu_count()
        self._log(f"🖥️  Dostępne rdzenie CPU: {cpu_count}")
        
        t_start = time.perf_counter()
        after_detailed = process_text_tokenized(original_text, after_regex, KEEP_LABELS)
        self.timing.detailed_labels_time = time.perf_counter() - t_start
        
        results['after_detailed_labels'] = after_detailed
        self._log(f"⏱️  Czas Detailed Labels: {self.timing.detailed_labels_time:.3f}s")
        self._log(f"Wynik Detailed Labels:\n{after_detailed[:300]}..." if len(after_detailed) > 300 else f"Wynik Detailed Labels:\n{after_detailed}")
        
        # === ETAP 4: Synthetic Generator ===
        self._log("\n" + "="*50)
        self._log("🔹 ETAP 4: Generacja danych syntetycznych")
        self._log("="*50)
        
        t_start = time.perf_counter()
        synthetic = generate_synthetic_output(after_detailed)
        self.timing.synthetic_generation_time = time.perf_counter() - t_start
        
        results['synthetic'] = synthetic
        self._log(f"⏱️  Czas Synthetic: {self.timing.synthetic_generation_time:.3f}s")
        self._log(f"Wynik Syntetyczny:\n{synthetic[:300]}..." if len(synthetic) > 300 else f"Wynik Syntetyczny:\n{synthetic}")
        
        # === ZAPIS DO synthetic_generation_Overfitters.txt ===
        t_io_start = time.perf_counter()
        self._log(f"\n💾 Zapisuję do {output_synth_path}...")
        with open(output_synth_path, 'w', encoding='utf-8') as f:
            f.write(synthetic)
        t_io_synth = time.perf_counter() - t_io_start
        
        self.timing.file_io_time = t_io_anon + t_io_synth
        
        # Czasy końcowe (bez ładowania modelu)
        self.timing.time_to_synthetic = time.perf_counter() - pipeline_start
        self.timing.total_time = self.timing.time_to_synthetic
        
        self._log(f"✅ Zapisano: {output_synth_path}")
        
        # Podsumowanie czasów
        self._log(str(self.timing))
        
        results['timing'] = self.timing
        
        return results
    
    def process_file(
        self,
        input_file: str,
        output_anonymized: str = OUTPUT_ANONYMIZED,
        output_synthetic: str = OUTPUT_SYNTHETIC
    ) -> dict:
        """
        Przetwarza plik przez cały pipeline.
        """
        self._log(f"📂 Wczytuję plik: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        return self.process(original_text, output_anonymized, output_synthetic)


# === BLOK URUCHAMIAJĄCY ===

def main():
    """Główna funkcja uruchamiająca pipeline."""
    
    print("\n" + "="*60)
    print("   🔐 PIPELINE ANONIMIZACJI I GENERACJI SYNTETYCZNEJ")
    print("   📊 Z mierzeniem czasu (bez ładowania modelu)")
    print("="*60)
    
    # Tryb interaktywny lub z pliku
    if len(sys.argv) >= 2:
        # Tryb plikowy
        input_file = sys.argv[1]
        output_anon = sys.argv[2] if len(sys.argv) >= 3 else OUTPUT_ANONYMIZED
        output_synth = sys.argv[3] if len(sys.argv) >= 4 else OUTPUT_SYNTHETIC
        
        pipeline = AnonymizationPipeline(verbose=True)
        pipeline.process_file(input_file, output_anon, output_synth)
    
    else:
        # Tryb interaktywny
        print("\nTryb interaktywny. Wpisz tekst do anonimizacji (lub 'q' by wyjść).")
        print("Możesz też uruchomić: python -m overfitters_pipeline.pipeline <plik_wejściowy>")
        print(f"Pliki wyjściowe zapisywane do: {OUTPUT_DIR}/")
        print("-" * 60)
        
        pipeline = AnonymizationPipeline(verbose=True)
        pipeline.load_models()
        
        while True:
            try:
                print("\n📝 Wprowadź tekst (lub 'q' by wyjść):")
                text = input("> ")
            except EOFError:
                break
            
            if text.lower() in ['q', 'exit', 'quit']:
                print("👋 Do widzenia!")
                break
            
            if not text.strip():
                continue
            
            results = pipeline.process(text)


if __name__ == "__main__":
    main()
