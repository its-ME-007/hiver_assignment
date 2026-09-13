"""Main entry point for Phase 1 data pipeline."""
import sys
from pathlib import Path
import pandas as pd
from .config_loader import load_config
from .logger import setup_logger, Metrics
from .thread_finder import ThreadFinder
from .thread_reconstructor import ThreadReconstructor
from .resolution_classifier import ResolutionClassifier
from .boilerplate_detector import BoilerplateDetector
from .text_normalizer import TextNormalizer, LanguageDetector
from .deduplicator import Deduplicator
from .data_exporter import DataExporter
from .validator import Validator


def run_phase1(config_path: str = None, input_csv: str = None) -> dict:
    """
    Run complete Phase 1 data cleaning pipeline.
    
    Args:
        config_path: Path to config.yaml
        input_csv: Path to input CSV (overrides config)
    
    Returns:
        Dictionary with pipeline results
    """
    # Load configuration
    config = load_config(config_path)
    logger = setup_logger(config)
    metrics = Metrics()
    
    logger.info("=" * 60)
    logger.info("Phase 1 Uber Support Data Pipeline Starting")
    logger.info("=" * 60)
    
    # Override input CSV if provided
    if input_csv:
        config["INPUT_CSV"] = input_csv
    
    input_path = config["INPUT_CSV"]
    
    try:
        # Step 1: Load dataset
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        metrics.set("initial_row_count", len(df))
        logger.info(f"Loaded {len(df)} rows from dataset")
        
        # Step 2: Discover threads
        logger.info("Step 1: Thread Discovery")
        finder = ThreadFinder(config, logger)
        root_ids, tweet_lookup = finder.find_uber_threads(df)
        metrics.set("root_threads_found", len(root_ids))
        
        # Step 3: Normalize text
        logger.info("Step 2: Text Normalization")
        normalizer = TextNormalizer(logger)
        for tweet_id, tweet in tweet_lookup.items():
            tweet.normalized_text = normalizer.normalize_text(tweet.text)
        
        # Step 4: Language detection
        language_filtering = config.get("LANGUAGE_FILTERING", False)
        if language_filtering:
            logger.info("Step 3: Language Detection")
            lang_detector = LanguageDetector(config, logger)
            for tweet_id, tweet in tweet_lookup.items():
                tweet.language = lang_detector.detect_language(tweet.normalized_text)
        else:
            logger.info("Step 3: Language Detection (disabled via config, defaulting to 'en')")
            for tweet_id, tweet in tweet_lookup.items():
                tweet.language = "en"
        
        # Step 5: Reconstruct threads
        logger.info("Step 4: Thread Reconstruction")
        reconstructor = ThreadReconstructor(config, logger, metrics)
        threads = reconstructor.reconstruct_all(root_ids, tweet_lookup)
        logger.info(f"Reconstructed {len(threads)} threads")

        # Step 4.5: Language Filtering (thread-level, keyed on root customer tweet)
        if language_filtering:
            logger.info("Step 4.5: Language Filtering")
            accepted_languages = set(config.get("ACCEPTED_LANGUAGES", ["en"]))
            pre_filter_count = len(threads)
            threads = [
                t for t in threads
                if t.tweets and t.tweets[0].language in accepted_languages
            ]
            filtered_count = pre_filter_count - len(threads)
            metrics.set("language_filtered_threads", filtered_count)
            logger.info(
                "Language filtering complete",
                pre_filter_count=pre_filter_count,
                post_filter_count=len(threads),
                filtered_out=filtered_count,
                accepted_languages=sorted(accepted_languages)
            )
        else:
            metrics.set("language_filtered_threads", 0)
        
        # Step 6: Classify resolution
        logger.info("Step 5: Resolution Classification")
        classifier = ResolutionClassifier(config, logger)
        threads = classifier.classify_all(threads)
        
        # Step 7: Detect boilerplate
        logger.info("Step 6: Boilerplate Detection")
        boilerplate_detector = BoilerplateDetector(config, logger, metrics, normalizer=normalizer)
        threads = boilerplate_detector.apply_boilerplate_flags(threads)
        
        # Step 8: Deduplication
        logger.info("Step 7: Deduplication")
        deduplicator = Deduplicator(logger, metrics)
        threads = deduplicator.deduplicate_threads(threads)
        metrics.set("threads_after_dedup", len(threads))
        
        # Step 9: Export to parquet
        logger.info("Step 8: Export to Parquet")
        exporter = DataExporter(config, logger, metrics)
        parquet_path = exporter.export_parquet(threads)  # deduplicated threads
        
        # Step 10: Generate decision log
        logger.info("Step 9: Generate Decision Log")
        decision_log_path = exporter.generate_decision_log()
        
        # Step 11: Validation
        logger.info("Step 10: Validation")
        df_output = pd.read_parquet(parquet_path)
        validator = Validator(config, logger)
        validation_report = validator.validate_output(df_output)
        
        logger.info("=" * 60)
        logger.info(f"Phase 1 Complete - Status: {validation_report.status}")
        logger.info("=" * 60)
        
        # Print metrics report
        logger.info(metrics.report())
        
        return {
            "status": "success",
            "validation_status": validation_report.status,
            "parquet_path": parquet_path,
            "decision_log_path": decision_log_path,
            "metrics": metrics.to_dict(),
            "validation_report": validation_report.to_dict(),
            "threads_count": len(threads),
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "metrics": metrics.to_dict(),
        }


if __name__ == "__main__":
    config_path = None
    input_csv = None
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        input_csv = sys.argv[2]
    
    result = run_phase1(config_path, input_csv)
    
    if result["status"] == "success":
        print(f"\n✓ Phase 1 completed successfully!")
        print(f"  Parquet: {result['parquet_path']}")
        print(f"  Decision Log: {result['decision_log_path']}")
        print(f"  Validation: {result['validation_status']}")
        sys.exit(0)
    else:
        print(f"\n✗ Phase 1 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)"""Main entry point for Phase 1 data pipeline."""
import sys
from pathlib import Path
import pandas as pd
from .config_loader import load_config
from .logger import setup_logger, Metrics
from .thread_finder import ThreadFinder
from .thread_reconstructor import ThreadReconstructor
from .resolution_classifier import ResolutionClassifier
from .boilerplate_detector import BoilerplateDetector
from .text_normalizer import TextNormalizer, LanguageDetector
from .deduplicator import Deduplicator
from .data_exporter import DataExporter
from .validator import Validator


def run_phase1(config_path: str = None, input_csv: str = None) -> dict:
    """
    Run complete Phase 1 data cleaning pipeline.
    
    Args:
        config_path: Path to config.yaml
        input_csv: Path to input CSV (overrides config)
    
    Returns:
        Dictionary with pipeline results
    """
    # Load configuration
    config = load_config(config_path)
    logger = setup_logger(config)
    metrics = Metrics()
    
    logger.info("=" * 60)
    logger.info("Phase 1 Uber Support Data Pipeline Starting")
    logger.info("=" * 60)
    
    # Override input CSV if provided
    if input_csv:
        config["INPUT_CSV"] = input_csv
    
    input_path = config["INPUT_CSV"]
    
    try:
        # Step 1: Load dataset
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        metrics.set("initial_row_count", len(df))
        logger.info(f"Loaded {len(df)} rows from dataset")
        
        # Step 2: Discover threads
        logger.info("Step 1: Thread Discovery")
        finder = ThreadFinder(config, logger)
        root_ids, tweet_lookup = finder.find_uber_threads(df)
        metrics.set("root_threads_found", len(root_ids))
        
        # Step 3: Normalize text
        logger.info("Step 2: Text Normalization")
        normalizer = TextNormalizer(logger)
        for tweet_id, tweet in tweet_lookup.items():
            tweet.normalized_text = normalizer.normalize_text(tweet.text)
        
        # Step 4: Language detection
        language_filtering = config.get("LANGUAGE_FILTERING", False)
        if language_filtering:
            logger.info("Step 3: Language Detection")
            lang_detector = LanguageDetector(config, logger)
            for tweet_id, tweet in tweet_lookup.items():
                tweet.language = lang_detector.detect_language(tweet.normalized_text)
        else:
            logger.info("Step 3: Language Detection (disabled via config, defaulting to 'en')")
            for tweet_id, tweet in tweet_lookup.items():
                tweet.language = "en"
        
        # Step 5: Reconstruct threads
        logger.info("Step 4: Thread Reconstruction")
        reconstructor = ThreadReconstructor(config, logger, metrics)
        threads = reconstructor.reconstruct_all(root_ids, tweet_lookup)
        logger.info(f"Reconstructed {len(threads)} threads")

        # Step 4.5: Language Filtering (thread-level, keyed on root customer tweet)
        if language_filtering:
            logger.info("Step 4.5: Language Filtering")
            accepted_languages = set(config.get("ACCEPTED_LANGUAGES", ["en"]))
            pre_filter_count = len(threads)
            threads = [
                t for t in threads
                if t.tweets and t.tweets[0].language in accepted_languages
            ]
            filtered_count = pre_filter_count - len(threads)
            metrics.set("language_filtered_threads", filtered_count)
            logger.info(
                "Language filtering complete",
                pre_filter_count=pre_filter_count,
                post_filter_count=len(threads),
                filtered_out=filtered_count,
                accepted_languages=sorted(accepted_languages)
            )
        else:
            metrics.set("language_filtered_threads", 0)
        
        # Step 6: Classify resolution
        logger.info("Step 5: Resolution Classification")
        classifier = ResolutionClassifier(config, logger)
        threads = classifier.classify_all(threads)
        
        # Step 7: Detect boilerplate
        logger.info("Step 6: Boilerplate Detection")
        boilerplate_detector = BoilerplateDetector(config, logger, metrics, normalizer=normalizer)
        threads = boilerplate_detector.apply_boilerplate_flags(threads)
        
        # Step 8: Deduplication
        logger.info("Step 7: Deduplication")
        deduplicator = Deduplicator(logger, metrics)
        threads = deduplicator.deduplicate_threads(threads)
        metrics.set("threads_after_dedup", len(threads))
        
        # Step 9: Export to parquet
        logger.info("Step 8: Export to Parquet")
        exporter = DataExporter(config, logger, metrics)
        parquet_path = exporter.export_parquet(threads)  # deduplicated threads
        
        # Step 10: Generate decision log
        logger.info("Step 9: Generate Decision Log")
        decision_log_path = exporter.generate_decision_log()
        
        # Step 11: Validation
        logger.info("Step 10: Validation")
        df_output = pd.read_parquet(parquet_path)
        validator = Validator(config, logger)
        validation_report = validator.validate_output(df_output)
        
        logger.info("=" * 60)
        logger.info(f"Phase 1 Complete - Status: {validation_report.status}")
        logger.info("=" * 60)
        
        # Print metrics report
        logger.info(metrics.report())
        
        return {
            "status": "success",
            "validation_status": validation_report.status,
            "parquet_path": parquet_path,
            "decision_log_path": decision_log_path,
            "metrics": metrics.to_dict(),
            "validation_report": validation_report.to_dict(),
            "threads_count": len(threads),
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "metrics": metrics.to_dict(),
        }


if __name__ == "__main__":
    config_path = None
    input_csv = None
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        input_csv = sys.argv[2]
    
    result = run_phase1(config_path, input_csv)
    
    if result["status"] == "success":
        print(f"\n✓ Phase 1 completed successfully!")
        print(f"  Parquet: {result['parquet_path']}")
        print(f"  Decision Log: {result['decision_log_path']}")
        print(f"  Validation: {result['validation_status']}")
        sys.exit(0)
    else:
        print(f"\n✗ Phase 1 failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)