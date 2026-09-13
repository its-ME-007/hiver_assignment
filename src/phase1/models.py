"""Data models for Phase 1 pipeline: Tweet and Thread classes."""
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Tweet:
    """Represents a single tweet in the dataset."""
    
    tweet_id: int
    author_id: str
    inbound: bool
    text: str
    original_text: str
    normalized_text: str
    language: str
    created_at: str
    timestamp: int
    in_response_to_tweet_id: Optional[int] = None
    response_tweet_id: Optional[int] = None
    is_boilerplate: bool = False
    
    def is_valid(self) -> bool:
        """
        Validate tweet has all required fields non-null.
        
        Returns:
            True if tweet is valid, False otherwise
        """
        required_fields = [
            self.tweet_id,
            self.author_id,
            self.text,
            self.original_text,
            self.normalized_text,
            self.language,
            self.created_at,
            self.timestamp
        ]
        return all(field is not None for field in required_fields)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tweet to dictionary for serialization.
        
        Returns:
            Dictionary representation of tweet
        """
        return asdict(self)
    
    def to_parquet_row(self) -> Dict[str, Any]:
        """
        Convert tweet to parquet-compatible row format.
        
        Returns:
            Dictionary with parquet-compatible types
        """
        return {
            "tweet_id": self.tweet_id,
            "author_id": self.author_id,
            "inbound": self.inbound,
            "text": self.text,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "language": self.language,
            "created_at": self.created_at,
            "timestamp": self.timestamp,
            "in_response_to_tweet_id": self.in_response_to_tweet_id,
            "response_tweet_id": self.response_tweet_id,
            "is_boilerplate": self.is_boilerplate,
        }
    
    def __repr__(self) -> str:
        """Human-readable representation of tweet."""
        return (
            f"Tweet(id={self.tweet_id}, author={self.author_id}, "
            f"inbound={self.inbound}, text_len={len(self.text)}, "
            f"is_boilerplate={self.is_boilerplate})"
        )


@dataclass
class Thread:
    """Represents a complete conversation thread."""
    
    thread_id: str
    root_tweet_id: int
    tweets: List[Tweet]
    turn_count: int
    dropped_branches: List[int] = field(default_factory=list)
    dropped_branch_count: int = 0
    resolution_tier: str = "unresolved_or_ongoing"
    is_boilerplate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate_schema(self) -> bool:
        """
        Validate thread schema matches specification.
        
        Returns:
            True if thread is valid, False otherwise
        
        Raises:
            TypeError: If schema doesn't match specification
        """
        # Check all required fields
        if not isinstance(self.thread_id, str):
            raise TypeError(f"thread_id must be str, got {type(self.thread_id)}")
        if not isinstance(self.root_tweet_id, int):
            raise TypeError(f"root_tweet_id must be int, got {type(self.root_tweet_id)}")
        if not isinstance(self.tweets, list):
            raise TypeError(f"tweets must be list, got {type(self.tweets)}")
        if not isinstance(self.turn_count, int):
            raise TypeError(f"turn_count must be int, got {type(self.turn_count)}")
        if self.turn_count < 1:
            raise TypeError(f"turn_count must be >= 1, got {self.turn_count}")
        if not isinstance(self.dropped_branch_count, int):
            raise TypeError(f"dropped_branch_count must be int, got {type(self.dropped_branch_count)}")
        if self.dropped_branch_count < 0:
            raise TypeError(f"dropped_branch_count must be >= 0, got {self.dropped_branch_count}")
        if self.resolution_tier not in ["resolved_explicit", "resolved_implicit", "unresolved_or_ongoing"]:
            raise TypeError(f"resolution_tier must be one of 3 values, got {self.resolution_tier}")
        if not isinstance(self.is_boilerplate, bool):
            raise TypeError(f"is_boilerplate must be bool, got {type(self.is_boilerplate)}")
        
        return True
    
    def to_parquet_row(self) -> Dict[str, Any]:
        """
        Convert thread to parquet-compatible row format.
        
        Returns:
            Dictionary with parquet output schema
        """
        # Get customer text (from first/root tweet)
        customer_text = self.tweets[0].normalized_text if self.tweets else ""
        
        # Get final brand reply (last tweet if brand, or second-to-last if customer)
        brand_reply = ""
        if len(self.tweets) >= 2:
            # Find last brand reply
            for tweet in reversed(self.tweets):
                if not tweet.inbound:  # Is brand reply
                    brand_reply = tweet.normalized_text
                    break
        
        # Convert timestamp - first tweet's timestamp
        timestamp = self.tweets[0].timestamp if self.tweets else 0
        
        return {
            "thread_id": self.thread_id,
            "customer_text": customer_text,
            "brand_reply": brand_reply,
            "resolution_tier": self.resolution_tier,
            "is_boilerplate": self.is_boilerplate,
            "turn_count": self.turn_count,
            "timestamp": timestamp,
            "dropped_branch_count": self.dropped_branch_count,
        }
    
    def __len__(self) -> int:
        """Return thread length (turn count)."""
        return self.turn_count
    
    def __repr__(self) -> str:
        """Human-readable representation of thread."""
        return (
            f"Thread(id={self.thread_id}, turns={self.turn_count}, "
            f"tier={self.resolution_tier}, boilerplate={self.is_boilerplate}, "
            f"dropped_branches={self.dropped_branch_count})"
        )


class ParquetSchema:
    """Output parquet schema definition and validation."""
    
    SCHEMA = {
        "thread_id": str,
        "customer_text": str,
        "brand_reply": str,
        "resolution_tier": str,
        "is_boilerplate": bool,
        "turn_count": int,
        "timestamp": int,
        "dropped_branch_count": int,
    }
    
    VALID_RESOLUTION_TIERS = {
        "resolved_explicit",
        "resolved_implicit",
        "unresolved_or_ongoing"
    }
    
    @classmethod
    def validate_row(cls, row: Dict[str, Any]) -> bool:
        """
        Validate a single row against schema.
        
        Args:
            row: Dictionary to validate
        
        Returns:
            True if valid, raises TypeError otherwise
        """
        # Check all required keys present
        missing_keys = set(cls.SCHEMA.keys()) - set(row.keys())
        if missing_keys:
            raise TypeError(f"Missing required columns: {missing_keys}")
        
        # Check data types
        for key, expected_type in cls.SCHEMA.items():
            actual_type = type(row[key])
            
            # Special handling for integer/float/numpy types
            if expected_type == int:
                if not isinstance(row[key], (int, type(None))):
                    if actual_type.__name__ in ("int64", "int32", "int16"):
                        continue  # Accept numpy integers
                    raise TypeError(f"Column {key}: expected int, got {actual_type}")
            elif expected_type == bool:
                if not isinstance(row[key], bool):
                    raise TypeError(f"Column {key}: expected bool, got {actual_type}")
            elif expected_type == str:
                if not isinstance(row[key], str):
                    raise TypeError(f"Column {key}: expected str, got {actual_type}")
        
        # Validate resolution_tier is one of allowed values
        if row["resolution_tier"] not in cls.VALID_RESOLUTION_TIERS:
            raise TypeError(
                f"resolution_tier must be one of {cls.VALID_RESOLUTION_TIERS}, "
                f"got {row['resolution_tier']}"
            )
        
        # Validate turn_count >= 1
        if row["turn_count"] < 1:
            raise TypeError(f"turn_count must be >= 1, got {row['turn_count']}")
        
        # Validate dropped_branch_count >= 0
        if row["dropped_branch_count"] < 0:
            raise TypeError(f"dropped_branch_count must be >= 0, got {row['dropped_branch_count']}")
        
        return True
    
    @classmethod
    def validate_dataframe(cls, df) -> bool:
        """
        Validate entire dataframe against schema.
        
        Args:
            df: Pandas DataFrame to validate
        
        Returns:
            True if valid, raises TypeError otherwise
        """
        import pandas as pd
        
        # Check all required columns present
        missing_cols = set(cls.SCHEMA.keys()) - set(df.columns)
        if missing_cols:
            raise TypeError(f"Missing required columns: {missing_cols}")
        
        # Validate each row
        for idx, row in df.iterrows():
            cls.validate_row(row.to_dict())
        
        return True
    
    @classmethod
    def get_dtypes(cls) -> Dict[str, str]:
        """
        Get dtype specification for parquet write.
        
        Returns:
            Dictionary of column names to dtype strings
        """
        return {
            "thread_id": "str",
            "customer_text": "str",
            "brand_reply": "str",
            "resolution_tier": "str",
            "is_boilerplate": "bool",
            "turn_count": "int64",
            "timestamp": "int64",
            "dropped_branch_count": "int64",
        }
