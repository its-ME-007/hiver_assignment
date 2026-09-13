"""Unit tests for data models."""
import pytest
from src.phase1.models import Tweet, Thread, ParquetSchema


class TestTweet:
    """Test Tweet model."""
    
    def test_tweet_creation(self):
        """Test creating a tweet."""
        tweet = Tweet(
            tweet_id=123,
            author_id="@user",
            inbound=True,
            text="Hello",
            original_text="Hello",
            normalized_text="hello",
            language="en",
            created_at="2026-01-01T00:00:00Z",
            timestamp=1000000
        )
        assert tweet.tweet_id == 123
        assert tweet.is_valid()
    
    def test_tweet_to_dict(self):
        """Test converting tweet to dict."""
        tweet = Tweet(
            tweet_id=123,
            author_id="@user",
            inbound=True,
            text="Hello",
            original_text="Hello",
            normalized_text="hello",
            language="en",
            created_at="2026-01-01T00:00:00Z",
            timestamp=1000000
        )
        d = tweet.to_dict()
        assert d["tweet_id"] == 123
        assert d["author_id"] == "@user"
    
    def test_tweet_to_parquet_row(self):
        """Test converting tweet to parquet row."""
        tweet = Tweet(
            tweet_id=123,
            author_id="@user",
            inbound=True,
            text="Hello",
            original_text="Hello",
            normalized_text="hello",
            language="en",
            created_at="2026-01-01T00:00:00Z",
            timestamp=1000000,
            is_boilerplate=False
        )
        row = tweet.to_parquet_row()
        assert row["tweet_id"] == 123
        assert row["is_boilerplate"] == False
    
    def test_tweet_repr(self):
        """Test tweet string representation."""
        tweet = Tweet(
            tweet_id=123,
            author_id="@user",
            inbound=True,
            text="Hello",
            original_text="Hello",
            normalized_text="hello",
            language="en",
            created_at="2026-01-01T00:00:00Z",
            timestamp=1000000
        )
        repr_str = repr(tweet)
        assert "123" in repr_str
        assert "@user" in repr_str


class TestThread:
    """Test Thread model."""
    
    def create_sample_thread(self):
        """Create a sample thread for testing."""
        tweet1 = Tweet(
            tweet_id=1,
            author_id="@customer",
            inbound=True,
            text="Help!",
            original_text="Help!",
            normalized_text="help",
            language="en",
            created_at="2026-01-01T00:00:00Z",
            timestamp=1000000
        )
        tweet2 = Tweet(
            tweet_id=2,
            author_id="@uber",
            inbound=False,
            text="We're here to help",
            original_text="We're here to help",
            normalized_text="we're here to help",
            language="en",
            created_at="2026-01-01T01:00:00Z",
            timestamp=1001000,
            in_response_to_tweet_id=1
        )
        
        thread = Thread(
            thread_id="thread_1",
            root_tweet_id=1,
            tweets=[tweet1, tweet2],
            turn_count=2,
            resolution_tier="resolved_implicit"
        )
        return thread
    
    def test_thread_creation(self):
        """Test creating a thread."""
        thread = self.create_sample_thread()
        assert thread.thread_id == "thread_1"
        assert len(thread.tweets) == 2
    
    def test_thread_validate_schema(self):
        """Test thread schema validation."""
        thread = self.create_sample_thread()
        assert thread.validate_schema()
    
    def test_thread_validate_schema_invalid_turn_count(self):
        """Test schema validation with invalid turn_count."""
        thread = self.create_sample_thread()
        thread.turn_count = 0
        with pytest.raises(TypeError):
            thread.validate_schema()
    
    def test_thread_validate_schema_invalid_resolution_tier(self):
        """Test schema validation with invalid resolution_tier."""
        thread = self.create_sample_thread()
        thread.resolution_tier = "invalid_tier"
        with pytest.raises(TypeError):
            thread.validate_schema()
    
    def test_thread_to_parquet_row(self):
        """Test converting thread to parquet row."""
        thread = self.create_sample_thread()
        row = thread.to_parquet_row()
        assert row["thread_id"] == "thread_1"
        assert row["turn_count"] == 2
        assert row["resolution_tier"] == "resolved_implicit"
        assert row["customer_text"] == "help"
    
    def test_thread_len(self):
        """Test thread length."""
        thread = self.create_sample_thread()
        assert len(thread) == 2
    
    def test_thread_repr(self):
        """Test thread string representation."""
        thread = self.create_sample_thread()
        repr_str = repr(thread)
        assert "thread_1" in repr_str
        assert "2" in repr_str


class TestParquetSchema:
    """Test ParquetSchema validation."""
    
    def test_schema_validate_valid_row(self):
        """Test validating a valid row."""
        row = {
            "thread_id": "thread_1",
            "customer_text": "Help",
            "brand_reply": "We help",
            "resolution_tier": "resolved_explicit",
            "is_boilerplate": False,
            "turn_count": 2,
            "timestamp": 1000000,
            "dropped_branch_count": 0,
        }
        assert ParquetSchema.validate_row(row)
    
    def test_schema_validate_missing_column(self):
        """Test validating row with missing column."""
        row = {
            "thread_id": "thread_1",
            "customer_text": "Help",
            "brand_reply": "We help",
            "resolution_tier": "resolved_explicit",
            # Missing is_boilerplate
            "turn_count": 2,
            "timestamp": 1000000,
            "dropped_branch_count": 0,
        }
        with pytest.raises(TypeError):
            ParquetSchema.validate_row(row)
    
    def test_schema_validate_invalid_resolution_tier(self):
        """Test validating row with invalid resolution_tier."""
        row = {
            "thread_id": "thread_1",
            "customer_text": "Help",
            "brand_reply": "We help",
            "resolution_tier": "invalid_tier",
            "is_boilerplate": False,
            "turn_count": 2,
            "timestamp": 1000000,
            "dropped_branch_count": 0,
        }
        with pytest.raises(TypeError):
            ParquetSchema.validate_row(row)
    
    def test_schema_validate_invalid_turn_count(self):
        """Test validating row with invalid turn_count."""
        row = {
            "thread_id": "thread_1",
            "customer_text": "Help",
            "brand_reply": "We help",
            "resolution_tier": "resolved_explicit",
            "is_boilerplate": False,
            "turn_count": 0,  # Invalid
            "timestamp": 1000000,
            "dropped_branch_count": 0,
        }
        with pytest.raises(TypeError):
            ParquetSchema.validate_row(row)
    
    def test_schema_dtypes(self):
        """Test getting dtype specification."""
        dtypes = ParquetSchema.get_dtypes()
        assert dtypes["turn_count"] == "int64"
        assert dtypes["is_boilerplate"] == "bool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
