"""
Indonesian Translation Model (Helsinki-NLP/opus-mt-en-id)
English → Indonesian translation wrapper using MarianMT.
"""

import logging
from typing import List, Dict, Any

import torch
from transformers import MarianTokenizer, MarianMTModel


logger = logging.getLogger(__name__)


class TranslationModel:
    """
    Thin wrapper around Helsinki-NLP/opus-mt-en-id for English → Indonesian translation.

    Key decisions:
    - CPU-only: MarianMT is small and fast on CPU
    - Batch-first design: One forward pass is much faster than looping
    - Graceful degradation: Return original English on error, never crash
    """

    MODEL_NAME = "Helsinki-NLP/opus-mt-en-id"

    def __init__(self):
        """Load the translation model and tokenizer from HuggingFace."""
        self._tokenizer: MarianTokenizer = None
        self._model: MarianMTModel = None
        self._is_loaded = False
        self._load_model()

    def _load_model(self):
        """Load MarianMT model and tokenizer."""
        try:
            self._tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = MarianMTModel.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            self._is_loaded = True
            logger.info(f"TranslationModel loaded: {self.MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load TranslationModel: {e}")
            raise

    def translate(self, text: str) -> str:
        """
        Translate a single English string to Indonesian.

        Args:
            text: English text to translate

        Returns:
            Indonesian translation, or original text if empty/whitespace
        """
        if not text or not text.strip():
            return ""

        try:
            return self._translate_text(text, max_length=512)[0]
        except Exception as e:
            logger.error(f"Translation failed for '{text}': {e}")
            return text  # Graceful degradation

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of English strings to Indonesian in a single forward pass.

        Args:
            texts: List of English strings to translate

        Returns:
            List of Indonesian translations in the same order; empty string for
            blank inputs; original texts on error.
        """
        if not texts:
            return []

        # Separate valid texts from blanks, preserving original positions
        indexed_valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results = [""] * len(texts)

        if not indexed_valid:
            return results

        valid_texts = [t for _, t in indexed_valid]

        try:
            tokenized = self._tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                translated_tokens = self._model.generate(
                    **tokenized, max_length=512, num_return_sequences=1
                )

            decoded = self._tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )

            # Place results back in original positions
            for (orig_idx, _), translation in zip(indexed_valid, decoded):
                results[orig_idx] = translation

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Graceful degradation: return originals for valid texts
            for orig_idx, orig_text in indexed_valid:
                results[orig_idx] = orig_text

        return results

    def _translate_text(self, text: str, **tokenizer_kwargs) -> List[str]:
        """
        Internal method to translate a single text.

        Args:
            text: English text to translate
            **tokenizer_kwargs: Passed to tokenizer

        Returns:
            List containing the translated text
        """
        tokenized = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=tokenizer_kwargs.get("max_length", 512),
            return_tensors="pt",
        )

        with torch.no_grad():
            translated_tokens = self._model.generate(
                **tokenized, max_length=512, num_return_sequences=1
            )

        decoded = self._tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        return decoded

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            Dict with model info
        """
        return {
            "name": self.MODEL_NAME,
            "is_loaded": self._is_loaded,
            "tokenizer_class": self._tokenizer.__class__.__name__,
            "model_class": self._model.__class__.__name__,
        }
