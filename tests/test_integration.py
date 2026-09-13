"""Integration tests for Phase 1 pipeline."""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.phase1.thread_finder import ThreadFinder
from src.phase1.thread_reconstructor import ThreadReconstructor
from src.phase1.resolution_classifier import ResolutionClassifier
from src.phase1.boilerplate_detector import BoilerplateDetector
from src.phase1.text_normalizer import TextNormalizer
from src.phase1.deduplicator import Deduplicator
from src.phase1.data_exporter import DataExporter
from src.phase1.validator import Validator
from src.phase1.models import Tweet, Thread
from src.phase1.logger import StructuredLogger, Metrics
from src.phase1.config_loader import load_config


class TestThreadDiscovery:
    """Test thread discovery from sample data."""
    
    def test_discover_threads_from_sample(self):
        """Test discovering threads from sample CSV."""
        config_path = Path(__file__).parent.parent / "src" / "phase1" / "config.yaml"
        config = load_config(str(config_path))
        logger = StructuredLogger("test")
        
        # Create sample data
        data = {
            'tweet_id': [1, 2, 3, 4],
            'author_id': ['@customer', 'Uber_Support', '@customer', 'Uber_Support'],
            'inbound': [True, False, True, False],
            'text': ['Help!', 'We help', 'Thanks!', 'Welcome'],
            'created_at': ['2026-01-01T00:00:00Z'] * 4,
            'timestamp': [1000000, 1000001, 1000002, 1000003],
            'in_response_to_tweet_id': [None, 1, 2, 3]
        }
        df = pd.DataFrame(data)
        
        # Discover threads
        finder = ThreadFinder(config, logger)
        root_ids, tweet_lookup = finder.find_uber_threads(df)
        
        # Verify results
        assert len(root_ids) > 0, "Should find at least one root thread"
        assert len(tweet_lookup) == 4, "Should have all tweets in lookup"


class TestThreadReconstruction:
    """Test thread reconstruction."""
    
    def test_reconstruct_simple_thread(self):
        """Test reconstructing a simple thread."""
        config = {
            "MAX_TURNS": 6,
            "ENFORCE_ALTERNATION": True,
        }
        logger = StructuredLogger("test")
        metrics = Metrics()
        
        # Create sample tweets
        tweets = {
            1: Tweet(tweet_id=1, author_id="@cust", inbound=True, text="Help",
                    original_text="Help", normalized_text="help", language="en",
                    created_at="2026-01-01T00:00:00Z", timestamp=1000000),
            2: Tweet(tweet_id=2, author_id="@uber", inbound=False, text="Help 2",
                    original_text="Help 2", normalized_text="help 2", language="en",
                    created_at="2026-01-01T01:00:00Z", timestamp=1001000,
                    in_response_to_tweet_id=1),
        }
        
        # Reconstruct
        reconstructor = ThreadReconstructor(config, logger, metrics)
        thread, dropped = reconstructor.build_thread(1, tweets)
        
        # Verify
        assert thread.root_tweet_id == 1
        assert len(thread.tweets) == 2
        assert thread.tweets[0].inbound == True
        assert thread.tweets[1].inbound == False


class TestEndToEnd:
    """End-to-end pipeline test."""
    
    def test_full_pipeline_with_sample_data(self):
        """Test complete pipeline with sample data."""
        config_path = Path(__file__).parent.parent / "src" / "phase1" / "config.yaml"
        config = load_config(str(config_path))
        logger = StructuredLogger("test_e2e")
        metrics = Metrics()
        
        # Create sample dataset
        data = {
            'tweet_id': list(range(1, 11)),
            'author_id': ['@customer', 'Uber_Support', '@customer', 'Uber_Support', 
                         '@customer', 'Uber_Support', '@customer', 'Uber_Support',
                         '@customer', 'Uber_Support'],
            'inbound': [True, False, True, False, True, False, True, False, True, False],
            'text': ['Need help', 'We help', 'Thanks!', 'You welcome',
                    'Issue resolved', 'Glad we helped', 'Bye', 'Goodbye',
                    'Final thanks', 'Final welcome'],
            'created_at': ['2026-01-01T00:00:00Z'] * 10,
            'timestamp': list(range(1000000, 1000010)),
            'in_response_to_tweet_id': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        df = pd.DataFrame(data)
        
        # Step 1: Discover
        finder = ThreadFinder(config, logger)
        root_ids, tweet_lookup = finder.find_uber_threads(df)
        assert len(root_ids) > 0, "Should find root threads"
        
        # Step 2: Normalize
        normalizer = TextNormalizer(logger)
        for tweet in tweet_lookup.values():
            tweet.normalized_text = normalizer.normalize_text(tweet.text)
        
        # Step 3: Reconstruct
        reconstructor = ThreadReconstructor(config, logger, metrics)
        threads = reconstructor.reconstruct_all(root_ids, tweet_lookup)
        assert len(threads) > 0, "Should reconstruct threads"
        
        # Step 4: Classify
        classifier = ResolutionClassifier(config, logger)
        threads = classifier.classify_all(threads)
        assert all(t.resolution_tier for t in threads), "Should classify all threads"
        
        # Step 5: Boilerplate
        boilerplate_detector = BoilerplateDetector(config, logger, metrics)
        threads = boilerplate_detector.apply_boilerplate_flags(threads)
        
        # Step 6: Export
        with tempfile.TemporaryDirectory() as tmpdir:
            config["OUTPUT_DIR"] = tmpdir
            config["OUTPUT_PARQUET"] = f"{tmpdir}/threads.parquet"
            config["DECISION_LOG"] = f"{tmpdir}/decision_log.md"
            
            exporter = DataExporter(config, logger, metrics)
            parquet_path = exporter.export_parquet(threads)
            
            # Verify parquet
            assert Path(parquet_path).exists(), "Parquet file should exist"
            df_out = pd.read_parquet(parquet_path)
            assert len(df_out) == len(threads), "All threads should be exported"
            
            # Validate
            validator = Validator(config, logger)
            report = validator.validate_output(df_out)
            assert report.schema_check["result"] == "PASS", "Schema should be valid"


class TestValidation:
    """Test validation of output."""
    
    def test_validation_with_sample_output(self):
        """Test validation of sample output."""
        config = {"PHASE0_PROJECTIONS": {"thread_count": 1000}}
        logger = StructuredLogger("test")
        
        # Create sample output
        data = {
            'thread_id': ['t1', 't2', 't3'],
            'customer_text': ['Help 1', 'Help 2', 'Help 3'],
            'brand_reply': ['Reply 1', 'Reply 2', 'Reply 3'],
            'resolution_tier': ['resolved_explicit', 'resolved_implicit', 'unresolved_or_ongoing'],
            'is_boilerplate': [False, False, True],
            'turn_count': [2, 3, 4],
            'timestamp': [1000000, 1000001, 1000002],
            'dropped_branch_count': [0, 1, 2],
        }
        df = pd.DataFrame(data)
        
        # Validate
        validator = Validator(config, logger)
        report = validator.validate_output(df, sample_size=2)
        
        # Check results
        assert report.schema_check["result"] == "PASS"
        assert len(report.sample_review["sampled_threads"]) == 2
        assert len(report.data_quality_issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
