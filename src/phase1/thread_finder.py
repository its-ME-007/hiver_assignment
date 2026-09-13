"""ThreadFinder: Discover Uber Support threads from raw dataset."""
import pandas as pd
from collections import defaultdict
from typing import Tuple, List, Dict, Any
from .models import Tweet
from .logger import StructuredLogger


class ThreadFinder:
    """Discover Uber Support customer-support threads from dataset."""
    
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        """
        Initialize ThreadFinder.
        
        Args:
            config: Configuration dictionary
            logger: StructuredLogger instance
        """
        self.config = config
        self.logger = logger
        self.uber_handles = set(config.get("UBER_HANDLES", ["Uber_Support"]))
        self.max_turns = config.get("MAX_TURNS", 6)
    
    def find_uber_threads(self, df: pd.DataFrame) -> Tuple[List[int], Dict[int, Tweet]]:
        """
        Discover all Uber Support threads from dataset.
        
        Algorithm:
        1. Identify Uber Support handle(s) via value_counts()
        2. Filter brand replies (inbound=False, author in UBER_HANDLES)
        3. For each brand reply, look up parent tweet
        4. Verify parent.inbound == True (customer)
        5. Build root_ids list and tweet_lookup dict
        
        Args:
            df: Raw tweet DataFrame with columns: tweet_id, author_id, inbound, text, etc.
        
        Returns:
            Tuple of (root_ids, tweet_lookup)
            - root_ids: List of root customer tweet IDs
            - tweet_lookup: Dict mapping tweet_id -> Tweet object
        
        Raises:
            ValueError: If no Uber handles found or dataset malformed
        """
        self.logger.info("Starting thread discovery")
        
        # Validate required columns (timestamp optional, will be generated)
        required_cols = [
            'tweet_id', 'author_id', 'inbound', 'text', 'created_at', 
            'in_response_to_tweet_id'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
        # Generate timestamp from created_at if not present
        if 'timestamp' not in df.columns:
            self.logger.debug("Generating timestamps from created_at")
            df = df.copy()
            try:
                # Explicit format avoids pandas falling back to the slow
                # per-row dateutil parser (the 'Could not infer format'
                # warning), which is a major slowdown at full dataset scale
                parsed = pd.to_datetime(
                    df['created_at'], format='%a %b %d %H:%M:%S %z %Y'
                )
            except (ValueError, TypeError):
                self.logger.warning(
                    "created_at did not match the expected Twitter format; "
                    "falling back to slower inferred parsing"
                )
                parsed = pd.to_datetime(df['created_at'])
            df['timestamp'] = parsed.astype('int64') // 10**9
        
        # Identify Uber Support handles
        handles_in_data = set(df[df['inbound'] == False]['author_id'].unique())
        uber_handles_found = handles_in_data & self.uber_handles
        
        if not uber_handles_found:
            raise ValueError(
                f"No Uber Support handles found in dataset. "
                f"Expected one of {self.uber_handles}, found handles: {handles_in_data}"
            )
        
        self.logger.info(
            f"Identified Uber Support handles",
            handles=str(list(uber_handles_found)),
            handle_count=len(uber_handles_found)
        )

        # --- Lightweight pass 1: find candidate roots without building Tweet
        # objects for the whole dataset. Uses plain dict/zip, not iterrows(),
        # and not the Tweet dataclass -- this only needs three columns. ---
        inbound_map = dict(zip(df['tweet_id'], df['inbound']))
        parent_map = dict(zip(df['tweet_id'], df['in_response_to_tweet_id']))

        brand_replies = df[
            (df['author_id'].isin(uber_handles_found)) &
            (df['inbound'] == False)
        ]
        brand_reply_count = len(brand_replies)

        root_ids = set()
        missing_parent_count = 0
        for tweet_id, parent_id in zip(brand_replies['tweet_id'], brand_replies['in_response_to_tweet_id']):
            if pd.isna(parent_id):
                continue
            parent_id = int(parent_id)
            if parent_id not in inbound_map:
                self.logger.warning(
                    "Parent tweet not found in dataset",
                    brand_reply_id=int(tweet_id),
                    parent_id=parent_id
                )
                missing_parent_count += 1
                continue
            if not inbound_map[parent_id]:
                self.logger.warning(
                    "Parent tweet is not from customer (inbound=False)",
                    brand_reply_id=int(tweet_id),
                    parent_id=parent_id
                )
                continue
            root_ids.add(parent_id)

        root_ids = sorted(root_ids)

        # --- Lightweight pass 2: BFS forward from each root, bounded to
        # MAX_TURNS hops (the reconstructor will never walk further than
        # that anyway), to find the small subset of tweet_ids that can
        # actually appear in a Uber thread. child_map is built from
        # in_response_to_tweet_id (not response_tweet_id) so it stays
        # consistent with how the reconstructor resolves replies. ---
        child_map: Dict[int, List[int]] = defaultdict(list)
        for tweet_id, parent_id in parent_map.items():
            if pd.notna(parent_id):
                child_map[int(parent_id)].append(tweet_id)

        relevant_ids = set(root_ids)
        frontier = set(root_ids)
        for _ in range(self.max_turns):
            next_frontier = set()
            for tid in frontier:
                for child_id in child_map.get(tid, []):
                    if child_id not in relevant_ids:
                        relevant_ids.add(child_id)
                        next_frontier.add(child_id)
            frontier = next_frontier
            if not frontier:
                break

        # --- Build Tweet objects only for the relevant subset. to_dict
        # ('records') is used instead of iterrows(), which materializes a
        # pandas Series (with dtype inference) per row and is one of the
        # more expensive things you can do in a pandas loop. ---
        self.logger.debug(
            f"Building tweet lookup for {len(relevant_ids)} relevant tweets "
            f"(out of {len(df)} total rows in dataset)"
        )
        relevant_df = df[df['tweet_id'].isin(relevant_ids)]

        tweet_lookup: Dict[int, Tweet] = {}
        for rec in relevant_df.to_dict('records'):
            tid = int(rec['tweet_id'])
            parent_id = rec.get('in_response_to_tweet_id')
            tweet_lookup[tid] = Tweet(
                tweet_id=tid,
                author_id=str(rec['author_id']),
                inbound=bool(rec['inbound']),
                text=str(rec['text']),
                original_text=str(rec['text']),
                normalized_text=str(rec['text']),  # Will be normalized later
                language="unknown",  # Will be detected later
                created_at=str(rec['created_at']),
                timestamp=int(rec['timestamp']),
                in_response_to_tweet_id=(int(parent_id) if pd.notna(parent_id) else None)
            )
        
        self.logger.info(
            "Thread discovery complete",
            brand_replies_found=brand_reply_count,
            root_threads_found=len(root_ids),
            missing_parents=missing_parent_count,
            tweets_indexed=len(tweet_lookup),
            total_dataset_rows=len(df)
        )
        
        return root_ids, tweet_lookup


def discover_threads(
    df: pd.DataFrame,
    config: Dict[str, Any],
    logger: StructuredLogger
) -> Tuple[List[int], Dict[int, Tweet]]:
    """
    Convenience function to discover threads.
    
    Args:
        df: Raw tweet DataFrame
        config: Configuration dictionary
        logger: StructuredLogger instance
    
    Returns:
        Tuple of (root_ids, tweet_lookup)
    """
    finder = ThreadFinder(config, logger)
    return finder.find_uber_threads(df)