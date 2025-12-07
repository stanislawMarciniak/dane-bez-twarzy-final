"""
Overfitters Pipeline - Zunifikowany system anonimizacji i generacji syntetycznej.

Moduły:
- regex_layer: Warstwa regex do wykrywania formatowanych danych
- detailed_labels: Dodawanie etykiet morfologicznych (płeć, przypadek)
- synthetic_generator: Generacja danych syntetycznych
- synthetic_data_pool: Pule danych syntetycznych
- morfeusz_inflector: Odmiana przez przypadki
- pipeline: Główny pipeline łączący wszystkie warstwy
"""

from .regex_layer import RegexLayer, DetectedEntity, EntityType
from .pipeline import AnonymizationPipeline

__all__ = [
    'RegexLayer',
    'DetectedEntity', 
    'EntityType',
    'AnonymizationPipeline',
]

