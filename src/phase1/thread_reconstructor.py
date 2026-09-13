"""ThreadReconstructor: Rebuild complete threads forward from root."""
from typing import Tuple, List, Dict, Any, Set
from collections import defaultdict
from .models import Tweet, Thread
from .logger import StructuredLogger, Metrics


class ThreadReconstructor:
    """Reconstruct complete conversation threads forward from root customer tweets."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: StructuredLogger, 
        metrics: Metrics
    ):
        """
        Initialize ThreadReconstructor.
        
        Args:
            config: Configuration dictionary
            logger: StructuredLogger instance
            metrics: Metrics tracker
        """
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.max_turns = config.get("MAX_TURNS", 6)
        self.enforce_alternation = config.get("ENFORCE_ALTERNATION", True)
    
    def _build_reply_index(self, tweet_lookup: Dict[int, Tweet]) -> Dict[int, List[int]]:
        """
        Build the reverse index: in_response_to_tweet_id -> [response_ids],
        sorted chronologically. Built ONCE per pipeline run (see
        reconstruct_all) rather than once per thread -- this used to be
        rebuilt from scratch inside build_thread for every single root,
        which is O(num_roots * len(tweet_lookup)) and was the dominant cost
        of the whole pipeline at real dataset scale.

        Args:
            tweet_lookup: Dict of all tweets

        Returns:
            Dict mapping tweet_id -> sorted list of tweet_ids that replied to it
        """
        reply_index: Dict[int, List[int]] = defaultdict(list)
        for tweet_id, tweet in tweet_lookup.items():
            if tweet.in_response_to_tweet_id is not None:
                reply_index[tweet.in_response_to_tweet_id].append(tweet_id)

        for parent_id in reply_index:
            reply_index[parent_id].sort()

        return reply_index

    def build_thread(
        self, 
        root_id: int, 
        tweet_lookup: Dict[int, Tweet],
        reply_index: Dict[int, List[int]]
    ) -> Tuple[Thread, List[int]]:
        """
        Reconstruct a conversation thread forward from root customer tweet.
        
        Algorithm:
        1. Start with root_tweet (customer)
        2. Find all replies to current tweet (via prebuilt reply_index)
        3. Filter to opposite inbound value (alternation)
        4. Select chronologically earliest (by tweet_id)
        5. Log unselected replies as dropped_branches
        6. Continue until no valid replies or natural termination
        
        Args:
            root_id: Root customer tweet ID
            tweet_lookup: Dict of all tweets
            reply_index: Prebuilt reverse index from _build_reply_index,
                shared across all calls to build_thread for this run
        
        Returns:
            Tuple of (thread, dropped_branch_ids)
        
        Raises:
            KeyError: If root_id not in tweet_lookup
            AssertionError: If root tweet is not from customer
        """
        # Validate root tweet exists
        if root_id not in tweet_lookup:
            raise KeyError(f"Root tweet ID {root_id} not found in tweet_lookup")
        
        root_tweet = tweet_lookup[root_id]
        
        # Validate root is from customer
        if not root_tweet.inbound:
            raise AssertionError(
                f"Root tweet {root_id} is not from customer (inbound={root_tweet.inbound})"
            )
        
        # Forward traversal
        threads = [root_tweet]
        dropped_branches = []
        current_inbound = root_tweet.inbound
        turn_count = 1
        
        while True:
            current_tweet = threads[-1]
            
            # Find all replies to current tweet
            possible_replies = reply_index.get(current_tweet.tweet_id, [])
            
            if not possible_replies:
                # No more replies - thread ends naturally
                break
            
            # Filter to opposite inbound value (alternation enforcement)
            if self.enforce_alternation:
                valid_replies = [
                    reply_id for reply_id in possible_replies
                    if tweet_lookup[reply_id].inbound != current_inbound
                ]
            else:
                valid_replies = possible_replies
            
            if not valid_replies:
                # No valid replies under alternation constraint
                break
            
            # Select chronologically earliest (first in sorted list)
            selected_reply_id = valid_replies[0]
            
            # Log unselected branches
            for unselected_id in valid_replies[1:]:
                dropped_branches.append(unselected_id)
                self.logger.log_decision(
                    decision=f"Branch discarded",
                    reason="branch_selection_rule",
                    tweet_id=unselected_id,
                    selected_id=selected_reply_id,
                    reason_detail="earliest reply selected"
                )
            
            # Append selected tweet
            selected_tweet = tweet_lookup[selected_reply_id]
            threads.append(selected_tweet)
            current_inbound = selected_tweet.inbound
            turn_count += 1
            
            # Enforce max_turns cap unconditionally. The previous condition
            # here required BOTH turn_count > max_turns AND no further
            # replies -- but "no further replies" is already caught by the
            # loop's top-of-iteration check, so that condition could never
            # actually fire and the cap was never enforced.
            if turn_count >= self.max_turns:
                self.metrics.increment("threads_capped_at_max_turns")
                self.logger.debug(
                    f"Thread reached max_turns cap, stopping traversal",
                    root_id=root_id,
                    max_turns=self.max_turns
                )
                break
        
        # Create thread object
        thread_id = f"thread_{root_id}_{root_tweet.timestamp}"
        thread = Thread(
            thread_id=thread_id,
            root_tweet_id=root_id,
            tweets=threads,
            turn_count=len(threads),
            dropped_branches=dropped_branches,
            dropped_branch_count=len(dropped_branches),
            metadata={
                "root_customer_text": threads[0].normalized_text,
                "final_brand_reply": (
                    threads[-1].normalized_text if not threads[-1].inbound else ""
                ),
                "final_author_inbound": threads[-1].inbound,
            }
        )
        
        # Track metrics
        self.metrics.increment("threads_reconstructed")
        self.metrics.increment("total_turns", turn_count)
        if len(dropped_branches) > 0:
            self.metrics.increment("threads_with_branches")
            self.metrics.increment("dropped_branches", len(dropped_branches))
        
        self.logger.debug(
            f"Reconstructed thread {thread_id}",
            turns=turn_count,
            dropped_branches=len(dropped_branches)
        )
        
        return thread, dropped_branches
    
    def reconstruct_all(
        self,
        root_ids: List[int],
        tweet_lookup: Dict[int, Tweet]
    ) -> List[Thread]:
        """
        Reconstruct all threads from root IDs.
        
        Args:
            root_ids: List of root customer tweet IDs
            tweet_lookup: Dict of all tweets
        
        Returns:
            List of reconstructed Thread objects
        """
        threads = []
        errors = 0

        # Build once, reuse for every root -- see _build_reply_index docstring
        reply_index = self._build_reply_index(tweet_lookup)
        
        for i, root_id in enumerate(root_ids):
            try:
                thread, dropped = self.build_thread(root_id, tweet_lookup, reply_index)
                threads.append(thread)
                
                if (i + 1) % 100 == 0:
                    self.logger.debug(f"Reconstructed {i + 1}/{len(root_ids)} threads")
            except Exception as e:
                self.logger.warning(
                    f"Failed to reconstruct thread",
                    root_id=root_id,
                    error=str(e)
                )
                errors += 1
        
        self.logger.info(
            "Thread reconstruction complete",
            total_threads=len(threads),
            failed=errors,
            success_rate=f"{100 * len(threads) / len(root_ids):.1f}%"
        )
        
        return threads