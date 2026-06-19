"""
morph_tagger.model - архитектура модели.

- MHAModel: морфологический классификатор на основе multi-head attention
- EncoderBlock: блок энкодера (MHA + Feed-Forward)
"""

from morph_tagger.model.model import MHAModel, EncoderBlock

__all__ = ['MHAModel', 'EncoderBlock']