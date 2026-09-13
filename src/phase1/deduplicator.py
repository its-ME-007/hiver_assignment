"""Deduplicator: Remove exact duplicate tweets."""
import pandas as pd
from typing import Dict, Any
from .logger import StructuredLogger, Metrics


class Deduplicator:
    """Remove exact duplicate tweets/threads and empty records.

    Two dedup entry points are provided:
    - deduplicate_threads: operates directly on Thread objects, keyed by
      (root tweet author_id, root tweet normalized customer_text). This is
      the one the main pipeline uses, since Thread.to_parquet_row() /
      ParquetSchema do not expose author_id or a raw normalized_text column
      to key on downstream.
    - deduplicate: a generic dataframe-based utility for any tabular data
      that does carry author_id + normalized_text columns directly (e.g.
      deduping the raw tweet-level table before thread reconstruction, if
      ever needed). Not used by the current pipeline.
    """
    
    def __init__(self, logger: StructuredLogger, metrics: Metrics):
        """
        Initialize Deduplicator.
        
        Args:
            logger: StructuredLogger instance
            metrics: Metrics tracker
        """
        self.logger = logger
        self.metrics = metrics

    def deduplicate_threads(self, threads: list) -> list:
        """
        Remove duplicate/empty threads based on the root customer tweet.

        Algorithm:
        1. Drop threads with no tweets, or an empty root normalized_text
        2. Dedup key: (root_tweet.author_id, root_tweet.normalized_text)
        3. Keep first occurrence (by list order), drop the rest

        Args:
            threads: List of Thread objects (post reconstruction/classification)

        Returns:
            Deduplicated list of Thread objects
        """
        self.logger.debug(f"Starting thread-level deduplication on {len(threads)} threads")

        original_count = len(threads)
        seen: set = set()
        deduped = []
        empty_count = 0
        duplicate_count = 0

        for thread in threads:
            if not thread.tweets:
                empty_count += 1
                continue

            root_tweet = thread.tweets[0]
            customer_text = (root_tweet.normalized_text or "").strip()

            if customer_text == "":
                empty_count += 1
                self.logger.log_decision(
                    decision="Dropped thread with empty customer_text",
                    reason="empty_text_removal",
                    thread_id=thread.thread_id
                )
                continue

            key = (root_tweet.author_id, customer_text)
            if key in seen:
                duplicate_count += 1
                self.logger.log_decision(
                    decision="Dropped duplicate thread",
                    reason="duplicate_removal",
                    thread_id=thread.thread_id,
                    key="author_id + normalized_text"
                )
                continue

            seen.add(key)
            deduped.append(thread)

        final_count = len(deduped)

        self.logger.info(
            "Thread-level deduplication complete",
            original_count=original_count,
            final_count=final_count,
            removed_total=original_count - final_count,
            empty_text_removed=empty_count,
            duplicates_removed=duplicate_count
        )

        self.metrics.increment("dedup_empty_removed", empty_count)
        self.metrics.increment("dedup_duplicates_removed", duplicate_count)

        return deduped
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate tweets and empty records.
        
        Algorithm:
        1. Remove tweets with empty normalized_text
        2. Identify exact duplicates (author_id + normalized_text)
        3. Keep first occurrence, remove rest
        4. Verify deduplication complete
        
        Deduplication Key: (author_id, normalized_text)
        - Keep first occurrence
        - Remove subsequent occurrences
        
        Args:
            df: DataFrame with tweets
        
        Returns:
            Deduplicated DataFrame
        """
        self.logger.debug(f"Starting deduplication on {len(df)} rows")
        
        original_count = len(df)
        
        # Step 1: Remove empty normalized_text
        empty_mask = df['normalized_text'].str.strip() == ""
        empty_count = empty_mask.sum()
        df = df[~empty_mask].copy()
        
        if empty_count > 0:
            self.logger.log_decision(
                decision="Removed empty normalized_text",
                reason="empty_text_removal",
                count=empty_count
            )
        
        # Step 2: Identify and remove duplicates
        duplicate_mask = df.duplicated(
            subset=['author_id', 'normalized_text'],
            keep='first'
        )
        duplicate_count = duplicate_mask.sum()
        df = df[~duplicate_mask].copy()
        
        if duplicate_count > 0:
            self.logger.log_decision(
                decision="Removed exact duplicates",
                reason="duplicate_removal",
                count=duplicate_count,
                key="author_id + normalized_text"
            )
        
        # Step 3: Verify deduplication
        final_count = len(df)
        removed_count = original_count - final_count
        expected_removed = empty_count + duplicate_count
        
        if removed_count == expected_removed:
            self.logger.info(
                "Deduplication complete",
                original_count=original_count,
                final_count=final_count,
                removed_total=removed_count,
                empty_text_removed=empty_count,
                duplicates_removed=duplicate_count
            )
        else:
            self.logger.warning(
                "Deduplication mismatch",
                removed_total=removed_count,
                expected_removed=expected_removed
            )
        
        # Track metrics
        self.metrics.increment("dedup_empty_removed", empty_count)
        self.metrics.increment("dedup_duplicates_removed", duplicate_count)
        
        return df.reset_index(drop=True)