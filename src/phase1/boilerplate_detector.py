"""BoilerplateDetector: Detect templated brand replies."""
import re
from collections import Counter
from typing import Dict, Any, List, Set
import pandas as pd
from .models import Tweet
from .logger import StructuredLogger, Metrics
from .text_normalizer import TextNormalizer


class BoilerplateDetector:
    """Detect templated and frequently repeated brand replies."""
    
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger, metrics: Metrics,
                 normalizer: "TextNormalizer" = None):
        """
        Initialize BoilerplateDetector.
        
        Args:
            config: Configuration dictionary
            logger: StructuredLogger instance
            metrics: Metrics tracker
            normalizer: Optional TextNormalizer instance (created internally if omitted).
                Reused here instead of duplicating normalization regexes, so the two
                normalization paths in the pipeline can't drift out of sync.
        """
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.normalizer = normalizer or TextNormalizer(logger)
        self.min_occurrences = config.get("BOILERPLATE_MIN_OCCURRENCES", 5)
        self.min_share = config.get("BOILERPLATE_MIN_SHARE", 0.005)  # 0.5%
    
    def detect_boilerplate(self, brand_replies: List[Tweet]) -> pd.Series:
        """
        Detect boilerplate (templated) brand replies using dual thresholds.
        
        Algorithm:
        1. Normalize all brand reply texts
        2. Count frequency of each unique text
        3. For each unique text:
           - is_boilerplate = (freq >= min_occurrences) AND (share >= min_share)
        4. Return Series indexed by tweet_id with boolean values
        
        Dual Threshold Logic:
        - BOTH thresholds must be met (AND operation, not OR)
        - min_occurrences: Absolute frequency filter
        - min_share: Relative percentage filter
        
        Args:
            brand_replies: List of brand Tweet objects
        
        Returns:
            pd.Series indexed by tweet_id with boolean boilerplate flags
        """
        if not brand_replies:
            return pd.Series(dtype=bool)
        
        self.logger.debug(f"Detecting boilerplate in {len(brand_replies)} brand replies")
        
        # Normalize all texts
        normalized_texts = [self.normalizer.normalize_for_dedup(tweet.normalized_text) for tweet in brand_replies]
        
        # Count frequencies
        text_counts = Counter(normalized_texts)
        total_replies = len(brand_replies)
        
        # Identify boilerplate signatures
        boilerplate_signatures = []
        for normalized_text, count in text_counts.items():
            share = count / total_replies
            is_boilerplate = (count >= self.min_occurrences) and (share >= self.min_share)
            
            if is_boilerplate:
                boilerplate_signatures.append((normalized_text, count, share))
                self.logger.log_decision(
                    decision="Boilerplate signature detected",
                    reason="dual_threshold_met",
                    count=count,
                    share=f"{100 * share:.2f}%",
                    min_count=self.min_occurrences,
                    min_share=f"{100 * self.min_share:.2f}%"
                )
        
        # Build result series
        result_dict = {}
        boilerplate_tweet_count = 0
        
        for tweet, normalized_text in zip(brand_replies, normalized_texts):
            count = text_counts[normalized_text]
            share = count / total_replies
            is_boilerplate = (count >= self.min_occurrences) and (share >= self.min_share)
            
            result_dict[tweet.tweet_id] = is_boilerplate
            if is_boilerplate:
                boilerplate_tweet_count += 1
        
        result_series = pd.Series(result_dict)
        
        boilerplate_rate = boilerplate_tweet_count / total_replies if total_replies > 0 else 0
        
        self.logger.info(
            "Boilerplate detection complete",
            unique_signatures=len(text_counts),
            boilerplate_signatures=len(boilerplate_signatures),
            boilerplate_replies=boilerplate_tweet_count,
            boilerplate_rate=f"{100 * boilerplate_rate:.2f}%",
            min_occurrences=self.min_occurrences,
            min_share=f"{100 * self.min_share:.2f}%"
        )
        
        # Track metrics
        self.metrics.set("boilerplate_signatures", len(boilerplate_signatures))
        self.metrics.set("boilerplate_replies", boilerplate_tweet_count)
        self.metrics.set("boilerplate_rate", boilerplate_rate)
        
        return result_series
    
    def apply_boilerplate_flags(self, threads: list) -> list:
        """
        Apply boilerplate flags to threads.
        
        For each thread, set is_boilerplate based on the final brand reply.
        
        Args:
            threads: List of Thread objects
        
        Returns:
            List of threads with boilerplate flags set
        """
        # Collect all brand replies from all threads
        brand_replies = []
        thread_brand_tweet_map = {}  # Map (thread_id, tweet_id) -> thread
        
        for thread in threads:
            for tweet in thread.tweets:
                if not tweet.inbound:  # Brand reply
                    brand_replies.append(tweet)
                    # Track which thread this brand reply belongs to
                    thread_brand_tweet_map[tweet.tweet_id] = thread
        
        if not brand_replies:
            self.logger.warning("No brand replies found in threads")
            return threads
        
        # Detect boilerplate
        boilerplate_series = self.detect_boilerplate(brand_replies)
        
        # Apply flags to tweets
        for tweet_id, is_boilerplate in boilerplate_series.items():
            if tweet_id in thread_brand_tweet_map:
                thread = thread_brand_tweet_map[tweet_id]
                # Find the tweet in the thread and mark it
                for tweet in thread.tweets:
                    if tweet.tweet_id == tweet_id:
                        tweet.is_boilerplate = is_boilerplate
        
        # Set thread is_boilerplate based on final brand reply
        for thread in threads:
            # Find last brand reply in thread
            for tweet in reversed(thread.tweets):
                if not tweet.inbound:  # Is brand reply
                    thread.is_boilerplate = tweet.is_boilerplate
                    break
        
        return threads