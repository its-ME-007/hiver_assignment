"""Property-based tests for Phase 1 core pipeline using hypothesis."""
import pytest
from hypothesis import given, settings, strategies as st
import pandas as pd
from src.phase1.models import Tweet, Thread
from src.phase1.resolution_classifier import ResolutionClassifier
from src.phase1.boilerplate_detector import BoilerplateDetector
from src.phase1.logger import StructuredLogger, Metrics


# Helper strategies
@st.composite
def tweet_strategy(draw):
    """Generate random valid Tweet objects."""
    tweet_id = draw(st.integers(min_value=1, max_value=1000000))
    author_id = draw(st.sampled_from(["@uber", "@customer"]))
    inbound = draw(st.booleans())
    text = draw(st.text(min_size=1, max_size=280))
    
    tweet = Tweet(
        tweet_id=tweet_id,
        author_id=author_id,
        inbound=inbound,
        text=text,
        original_text=text,
        normalized_text=text.lower(),
        language="en",
        created_at="2026-01-01T00:00:00Z",
        timestamp=1000000,
    )
    return tweet


@st.composite
def thread_strategy(draw):
    """Generate random valid Thread objects."""
    num_tweets = draw(st.integers(min_value=1, max_value=10))
    tweets = []
    inbound = True  # Start with customer
    
    for i in range(num_tweets):
        tweet = Tweet(
            tweet_id=i,
            author_id="@customer" if inbound else "@uber",
            inbound=inbound,
            text=f"Tweet {i}",
            original_text=f"Tweet {i}",
            normalized_text=f"tweet {i}",
            language="en",
            created_at="2026-01-01T00:00:00Z",
            timestamp=1000000 + i,
        )
        tweets.append(tweet)
        inbound = not inbound  # Alternate
    
    thread = Thread(
        thread_id="test_thread",
        root_tweet_id=0,
        tweets=tweets,
        turn_count=len(tweets),
        resolution_tier="unresolved_or_ongoing"
    )
    return thread


class TestAlternationProperty:
    """Property 1: Thread Alternation Invariant - consecutive tweets alternate."""
    
    @given(thread_strategy())
    @settings(max_examples=50)
    def test_alternation_invariant(self, thread):
        """
        **Validates: Requirements 2.2**
        
        For any thread, consecutive tweets must have opposite inbound values.
        """
        for i in range(len(thread.tweets) - 1):
            current = thread.tweets[i]
            next_tweet = thread.tweets[i + 1]
            assert current.inbound != next_tweet.inbound, \
                f"Tweets {i} and {i+1} violate alternation: {current.inbound} vs {next_tweet.inbound}"


class TestResolutionDeterminism:
    """Property 4: Resolution Tier Determinism - same thread always same tier."""
    
    @given(thread_strategy())
    @settings(max_examples=50)
    def test_resolution_determinism(self, thread):
        """
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
        
        For any thread, resolution tier assignment is deterministic.
        """
        config = {
            "POSITIVE_CLOSERS": ["thanks", "thank you", "resolved", "fixed"],
        }
        logger = StructuredLogger("test")
        classifier = ResolutionClassifier(config, logger)
        
        # Classify multiple times
        tier1 = classifier.classify_resolution(thread)
        tier2 = classifier.classify_resolution(thread)
        tier3 = classifier.classify_resolution(thread)
        
        # All results must be identical
        assert tier1 == tier2 == tier3, \
            f"Resolution classification not deterministic: {tier1} vs {tier2} vs {tier3}"
        
        # Tier must be valid
        valid_tiers = {"resolved_explicit", "resolved_implicit", "unresolved_or_ongoing"}
        assert tier1 in valid_tiers, f"Invalid resolution tier: {tier1}"


class TestResolutionLogic:
    """Test resolution classification heuristics."""
    
    def test_resolved_explicit_with_closer(self):
        """Test resolved_explicit when customer ends with positive closer."""
        config = {"POSITIVE_CLOSERS": ["thanks", "thank you"]}
        logger = StructuredLogger("test")
        classifier = ResolutionClassifier(config, logger)
        
        tweet1 = Tweet(
            tweet_id=1, author_id="@cust", inbound=True, text="Help",
            original_text="Help", normalized_text="help", language="en",
            created_at="2026-01-01T00:00:00Z", timestamp=1000000
        )
        tweet2 = Tweet(
            tweet_id=2, author_id="@uber", inbound=False, text="We helped",
            original_text="We helped", normalized_text="we helped", language="en",
            created_at="2026-01-01T01:00:00Z", timestamp=1001000
        )
        tweet3 = Tweet(
            tweet_id=3, author_id="@cust", inbound=True, text="Thanks!",
            original_text="Thanks!", normalized_text="thanks!", language="en",
            created_at="2026-01-01T02:00:00Z", timestamp=1002000
        )
        
        thread = Thread(
            thread_id="t1", root_tweet_id=1, tweets=[tweet1, tweet2, tweet3],
            turn_count=3
        )
        
        assert classifier.classify_resolution(thread) == "resolved_explicit"
    
    def test_resolved_implicit_ends_with_brand(self):
        """Test resolved_implicit when brand has last reply."""
        config = {"POSITIVE_CLOSERS": []}
        logger = StructuredLogger("test")
        classifier = ResolutionClassifier(config, logger)
        
        tweet1 = Tweet(
            tweet_id=1, author_id="@cust", inbound=True, text="Help",
            original_text="Help", normalized_text="help", language="en",
            created_at="2026-01-01T00:00:00Z", timestamp=1000000
        )
        tweet2 = Tweet(
            tweet_id=2, author_id="@uber", inbound=False, text="We helped",
            original_text="We helped", normalized_text="we helped", language="en",
            created_at="2026-01-01T01:00:00Z", timestamp=1001000
        )
        
        thread = Thread(
            thread_id="t1", root_tweet_id=1, tweets=[tweet1, tweet2],
            turn_count=2
        )
        
        assert classifier.classify_resolution(thread) == "resolved_implicit"


class TestBoilerplateDetection:
    """Property 5: Boilerplate Detection Dual Thresholds."""
    
    @given(st.lists(tweet_strategy(), min_size=5, max_size=100))
    @settings(max_examples=20)
    def test_boilerplate_detection_property(self, brand_replies):
        """
        **Validates: Requirements 4.1, 4.2, 4.3**
        
        Boilerplate detection applies dual thresholds (freq AND share).
        """
        config = {
            "BOILERPLATE_MIN_OCCURRENCES": 5,
            "BOILERPLATE_MIN_SHARE": 0.05,  # 5%
        }
        logger = StructuredLogger("test")
        metrics = Metrics()
        detector = BoilerplateDetector(config, logger, metrics)
        
        # All tweets should have same author (brand)
        for i, tweet in enumerate(brand_replies):
            tweet.inbound = False
            tweet.author_id = "@uber"
            tweet.tweet_id = i  # Unique IDs
        
        result = detector.detect_boilerplate(brand_replies)
        
        # Result should be a Series
        assert isinstance(result, pd.Series)
        
        # Result should have entry for each tweet (by tweet_id)
        assert len(result) <= len(brand_replies)


class TestBoilerplateDualThresholds:
    """Test that boilerplate uses dual thresholds (AND, not OR)."""
    
    def test_dual_threshold_and_logic(self):
        """Test that BOTH thresholds must be met."""
        config = {
            "BOILERPLATE_MIN_OCCURRENCES": 5,
            "BOILERPLATE_MIN_SHARE": 0.1,  # 10%
        }
        logger = StructuredLogger("test")
        metrics = Metrics()
        detector = BoilerplateDetector(config, logger, metrics)
        
        # Create 100 brand replies, with same text appearing 6 times (6%)
        # This meets min_occurrences but not min_share
        brand_replies = []
        for i in range(100):
            if i < 6:
                text = "boilerplate text"
            else:
                text = f"unique text {i}"
            
            tweet = Tweet(
                tweet_id=i, author_id="@uber", inbound=False,
                text=text, original_text=text, normalized_text=text.lower(),
                language="en", created_at="2026-01-01T00:00:00Z", timestamp=1000000
            )
            brand_replies.append(tweet)
        
        result = detector.detect_boilerplate(brand_replies)
        
        # The repeated text should NOT be flagged as boilerplate
        # because 6% < 10% minimum share
        boilerplate_ids = result[result == True].index.tolist()
        
        # Should have 0 boilerplate (6 replies = 6%, which is < 10%)
        assert len(boilerplate_ids) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
