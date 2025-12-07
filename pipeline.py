#!/usr/bin/env python3
"""
Wrapper dla overfitters_pipeline.

Użycie:
    python pipeline.py <plik_wejściowy> [plik_anon] [plik_synth]
    
Lub jako moduł:
    python -m overfitters_pipeline.pipeline <plik_wejściowy>
"""

import sys
import os

# Dodaj ścieżkę projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overfitters_pipeline.pipeline import AnonymizationPipeline, main, TimingResult

# Eksportuj klasy
__all__ = ['AnonymizationPipeline', 'TimingResult', 'main']

if __name__ == "__main__":
    main()
