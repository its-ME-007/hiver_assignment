"""TextNormalizer: Normalize tweet text for embedding and deduplication."""
import re
import html
from typing import Dict, Any
from .logger import StructuredLogger


class TextNormalizer:
    """Normalize tweet text for downstream processing."""
    
    def __init__(self, logger: StructuredLogger = None):
        """
        Initialize TextNormalizer.
        
        Args:
            logger: Optional StructuredLogger instance
        """
        self.logger = logger
    
    def normalize_text(self, text: str) -> str:
        """
        Full normalization pipeline for primary analysis.
        
        Transformations (in order):
        1. Unescape HTML entities
        2. Replace URLs -> <URL>
        3. Replace mentions -> <MENTION>
        4. Replace 6+ digit sequences -> <ID>
        5. Collapse whitespace
        6. Strip edges
        
        Preserves: emojis, capitalization, punctuation
        
        Args:
            text: Raw tweet text
        
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Unescape HTML entities
        text = html.unescape(text)
        
        # 2. Replace URLs
        text = re.sub(r'https?://\S+', '<URL>', text)
        
        # 3. Replace mentions
        text = re.sub(r'@\w+', '<MENTION>', text)
        
        # 4. Replace 6+ digit sequences
        text = re.sub(r'\d{6,}', '<ID>', text)
        
        # 5. Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # 6. Strip edges
        text = text.strip()
        
        return text
    
    def normalize_for_dedup(self, text: str) -> str:
        """
        Normalization for deduplication and boilerplate detection.
        
        Similar to normalize_text but also lowercases.
        
        Args:
            text: Raw tweet text
        
        Returns:
            Normalized text for deduplication
        """
        # First apply standard normalization
        text = self.normalize_text(text)
        
        # Then lowercase
        text = text.lower()
        
        return text
    
    def normalize_tweet(self, text: str) -> Dict[str, str]:
        """
        Normalize a tweet and return both versions.
        
        Args:
            text: Raw tweet text
        
        Returns:
            Dictionary with 'normalized_text' and 'normalized_for_dedup'
        """
        normalized = self.normalize_text(text)
        normalized_dedup = self.normalize_for_dedup(text)
        
        return {
            "normalized_text": normalized,
            "normalized_for_dedup": normalized_dedup
        }


class LanguageDetector:
    """Detect language of tweets using langdetect."""
    
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger = None):
        """
        Initialize LanguageDetector.
        
        Args:
            config: Configuration dictionary
            logger: Optional StructuredLogger instance
        """
        self.config = config
        self.logger = logger
        self.language_filtering = config.get("LANGUAGE_FILTERING", False)
        self.accepted_languages = set(config.get("ACCEPTED_LANGUAGES", ["en"]))
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text using langdetect.
        
        Args:
            text: Text to analyze
        
        Returns:
            Language code (e.g., "en", "es", "fr")
        """
        try:
            from langdetect import detect
            language = detect(text)
            return language
        except Exception as e:
            if self.logger:
                self.logger.debug(
                    f"Language detection failed, defaulting to 'unknown'",
                    error=str(e)
                )
            return "unknown"
    
    def is_english(self, text: str) -> bool:
        """
        Check if text is English.
        
        Args:
            text: Text to check
        
        Returns:
            True if English, False otherwise
        """
        language = self.detect_language(text)
        return language == "en"
    
    def is_accepted_language(self, text: str) -> bool:
        """
        Check if text is in accepted languages.
        
        Args:
            text: Text to check
        
        Returns:
            True if language is accepted, False otherwise
        """
        language = self.detect_language(text)
        return language in self.accepted_languages


def normalize_text(text: str) -> str:
    """
    Convenience function for text normalization.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text
    """
    normalizer = TextNormalizer()
    return normalizer.normalize_text(text)


def normalize_for_dedup(text: str) -> str:
    """
    Convenience function for deduplication normalization.
    
    Args:
        text: Text to normalize
    
    Returns:
        Normalized text for deduplication
    """
    normalizer = TextNormalizer()
    return normalizer.normalize_for_dedup(text)
