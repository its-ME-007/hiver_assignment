"""ResolutionClassifier: Classify thread resolution outcomes."""
import re
from typing import Dict, Any, Set
from .models import Thread
from .logger import StructuredLogger


class ResolutionClassifier:
    """Classify threads into three resolution tiers."""
    
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger, metrics=None):
        """
        Initialize ResolutionClassifier.
        
        Args:
            config: Configuration dictionary
            logger: StructuredLogger instance
            metrics: Optional Metrics tracker. Without it, resolution tier
                counts are only ever logged to console, not recorded --
                DataExporter's decision log reads these from metrics and
                falls back to 'N/A' if none was passed.
        """
        self.config = config
        self.logger = logger
        self.metrics = metrics
        
        # Load positive closers from config and normalize
        closers = config.get("POSITIVE_CLOSERS", [])
        self.positive_closers: Set[str] = {closer.lower().strip() for closer in closers}
    
    def classify_resolution(self, thread: Thread) -> str:
        """
        Classify thread's resolution status into three tiers.
        
        Heuristic:
        - IF last tweet is from customer AND contains positive closer -> resolved_explicit
        - IF last tweet is from brand -> resolved_implicit (medium confidence)
        - ELSE (customer last tweet, no closer) -> unresolved_or_ongoing
        
        Args:
            thread: Thread to classify
        
        Returns:
            Resolution tier: "resolved_explicit", "resolved_implicit", or "unresolved_or_ongoing"
        """
        if not thread.tweets:
            return "unresolved_or_ongoing"
        
        last_tweet = thread.tweets[-1]
        
        # If last tweet is from customer
        if last_tweet.inbound:
            # Check for positive closers
            if self._contains_positive_closer(last_tweet.normalized_text):
                return "resolved_explicit"
            else:
                return "unresolved_or_ongoing"
        else:
            # Last tweet is from brand
            self.logger.debug(
                "Thread classified as resolved_implicit",
                thread_id=thread.thread_id,
                confidence="medium"
            )
            return "resolved_implicit"
    
    def _contains_positive_closer(self, text: str) -> bool:
        """
        Check if text contains any positive closer phrases.
        
        Uses normalized text (lowercase) and word boundary matching.
        
        Args:
            text: Normalized text to check
        
        Returns:
            True if positive closer found, False otherwise
        """
        text_lower = text.lower()
        
        for closer in self.positive_closers:
            # Use word boundary matching for exact phrases
            pattern = r'\b' + re.escape(closer) + r'\b'
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def classify_all(self, threads: list) -> list:
        """
        Classify all threads and assign resolution tiers.
        
        Args:
            threads: List of Thread objects
        
        Returns:
            List of threads with resolution_tier assigned
        """
        resolved_explicit_count = 0
        resolved_implicit_count = 0
        unresolved_count = 0
        
        for thread in threads:
            tier = self.classify_resolution(thread)
            thread.resolution_tier = tier
            
            if tier == "resolved_explicit":
                resolved_explicit_count += 1
            elif tier == "resolved_implicit":
                resolved_implicit_count += 1
            else:
                unresolved_count += 1
        
        total = len(threads)
        self.logger.info(
            "Resolution classification complete",
            resolved_explicit=resolved_explicit_count,
            resolved_explicit_pct=f"{100 * resolved_explicit_count / total if total > 0 else 0:.1f}%",
            resolved_implicit=resolved_implicit_count,
            resolved_implicit_pct=f"{100 * resolved_implicit_count / total if total > 0 else 0:.1f}%",
            unresolved=unresolved_count,
            unresolved_pct=f"{100 * unresolved_count / total if total > 0 else 0:.1f}%"
        )

        if self.metrics is not None:
            self.metrics.set("resolution_explicit_count", resolved_explicit_count)
            self.metrics.set("resolution_implicit_count", resolved_implicit_count)
            self.metrics.set("resolution_unresolved_count", unresolved_count)
        
        return threads