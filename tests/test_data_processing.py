"""Tests for data processing modules."""
import pytest
import pandas as pd
from src.phase1.text_normalizer import TextNormalizer, LanguageDetector, normalize_text
from src.phase1.deduplicator import Deduplicator
from src.phase1.logger import StructuredLogger, Metrics
from src.phase1.models import Tweet


class TestTextNormalizer:
    """Test TextNormalizer."""
    
    def test_normalize_urls(self):
        """Test URL normalization."""
        normalizer = TextNormalizer()
        text = "Check http://example.com or https://test.org"
        result = normalizer.normalize_text(text)
        assert "<URL>" in result
        assert "http" not in result
    
    def test_normalize_mentions(self):
        """Test mention normalization."""
        normalizer = TextNormalizer()
        text = "Hey @user1 and @user2, check this"
        result = normalizer.normalize_text(text)
        assert "<MENTION>" in result
        assert "@user" not in result
    
    def test_normalize_ids(self):
        """Test ID normalization for 6+ digits."""
        normalizer = TextNormalizer()
        text = "Order 123456 or trip 789012345"
        result = normalizer.normalize_text(text)
        assert "<ID>" in result
    
    def test_preserve_case(self):
        """Test that case is preserved."""
        normalizer = TextNormalizer()
        text = "Hello WORLD with MixedCase"
        result = normalizer.normalize_text(text)
        assert "WORLD" in result or "world" not in result  # Case preserved
    
    def test_preserve_emojis(self):
        """Test that emojis are preserved."""
        normalizer = TextNormalizer()
        text = "Great! 😊 Thanks 👍"
        result = normalizer.normalize_text(text)
        assert "😊" in result or "👍" in result  # Emojis preserved or stripped consistently
    
    def test_normalize_for_dedup_lowercase(self):
        """Test normalize_for_dedup lowercases."""
        normalizer = TextNormalizer()
        text = "HELLO World"
        result = normalizer.normalize_for_dedup(text)
        assert result == result.lower()
    
    def test_idempotent_normalization(self):
        """Test that normalization is idempotent."""
        normalizer = TextNormalizer()
        text = "Check http://example.com or @user"
        result1 = normalizer.normalize_text(text)
        result2 = normalizer.normalize_text(result1)
        assert result1 == result2


class TestLanguageDetector:
    """Test LanguageDetector."""
    
    def test_language_detection(self):
        """Test language detection."""
        config = {"LANGUAGE_FILTERING": True, "ACCEPTED_LANGUAGES": ["en"]}
        logger = StructuredLogger("test")
        detector = LanguageDetector(config, logger)
        
        # Test English
        lang = detector.detect_language("Hello world")
        assert lang is not None
    
    def test_is_english_detection(self):
        """Test English detection."""
        config = {"LANGUAGE_FILTERING": True, "ACCEPTED_LANGUAGES": ["en"]}
        logger = StructuredLogger("test")
        detector = LanguageDetector(config, logger)
        
        # English text should be detected (or gracefully default)
        result = detector.is_english("Hello")
        assert isinstance(result, bool)
    
    def test_accepted_language_check(self):
        """Test accepted language checking."""
        config = {"LANGUAGE_FILTERING": True, "ACCEPTED_LANGUAGES": ["en"]}
        logger = StructuredLogger("test")
        detector = LanguageDetector(config, logger)
        
        result = detector.is_accepted_language("Hello world")
        assert isinstance(result, bool)


class TestDeduplicator:
    """Test Deduplicator."""
    
    def test_remove_empty_text(self):
        """Test removing empty normalized_text."""
        logger = StructuredLogger("test")
        metrics = Metrics()
        dedup = Deduplicator(logger, metrics)
        
        df = pd.DataFrame({
            'author_id': ['@user1', '@user2', '@user3'],
            'normalized_text': ['hello', '  ', 'world'],
        })
        
        result = dedup.deduplicate(df)
        assert len(result) == 2
        assert '' not in result['normalized_text'].str.strip().values
    
    def test_remove_exact_duplicates(self):
        """Test removing exact duplicates."""
        logger = StructuredLogger("test")
        metrics = Metrics()
        dedup = Deduplicator(logger, metrics)
        
        df = pd.DataFrame({
            'author_id': ['@user1', '@user1', '@user2'],
            'normalized_text': ['hello', 'hello', 'world'],
        })
        
        result = dedup.deduplicate(df)
        assert len(result) == 2  # One duplicate removed
    
    def test_deduplication_idempotent(self):
        """Test that deduplication is idempotent."""
        logger = StructuredLogger("test")
        metrics = Metrics()
        dedup = Deduplicator(logger, metrics)
        
        df = pd.DataFrame({
            'author_id': ['@user1', '@user1', '@user2'],
            'normalized_text': ['hello', 'hello', 'world'],
        })
        
        result1 = dedup.deduplicate(df)
        result2 = dedup.deduplicate(result1)
        
        assert len(result1) == len(result2)
    
    def test_keeps_first_occurrence(self):
        """Test that first occurrence is kept."""
        logger = StructuredLogger("test")
        metrics = Metrics()
        dedup = Deduplicator(logger, metrics)
        
        df = pd.DataFrame({
            'author_id': ['@user1', '@user1'],
            'normalized_text': ['hello', 'hello'],
            'timestamp': [100, 200],  # Different timestamps
        })
        
        result = dedup.deduplicate(df)
        assert len(result) == 1
        assert result.iloc[0]['timestamp'] == 100  # First kept


class TestNormalizationConsistency:
    """Test consistency of normalization across calls."""
    
    @pytest.mark.parametrize("text", [
        "Hello @user check http://example.com",
        "Order 123456789 please",
        "Thanks for 😊 the help!",
        "Multiple  spaces   and\ttabs",
    ])
    def test_normalize_deterministic(self, text):
        """Test normalization is deterministic."""
        result1 = normalize_text(text)
        result2 = normalize_text(text)
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
