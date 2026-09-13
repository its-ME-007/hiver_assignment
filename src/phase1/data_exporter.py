"""DataExporter: Export threads to parquet and generate decision log."""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .models import Thread, ParquetSchema
from .logger import StructuredLogger, Metrics


class DataExporter:
    """Export reconstructed threads to parquet and documentation."""
    
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger, metrics: Metrics):
        """
        Initialize DataExporter.
        
        Args:
            config: Configuration dictionary
            logger: StructuredLogger instance
            metrics: Metrics tracker
        """
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.output_dir = Path(config.get("OUTPUT_DIR", "data/processed"))
        self.output_parquet = config.get("OUTPUT_PARQUET", "data/processed/threads.parquet")
        self.decision_log_path = config.get("DECISION_LOG", "data/processed/decision_log.md")
    
    def export_parquet(self, threads: List[Thread]) -> str:
        """
        Export reconstructed threads to parquet file.
        
        Args:
            threads: List of Thread objects
        
        Returns:
            Path to exported parquet file
        
        Raises:
            TypeError: If schema validation fails
        """
        self.logger.debug(f"Preparing {len(threads)} threads for parquet export")
        
        # Build list of rows
        rows = []
        for thread in threads:
            row = thread.to_parquet_row()
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Validate schema
        self.logger.debug("Validating parquet schema")
        ParquetSchema.validate_dataframe(df)
        
        # Create output directory
        output_path = Path(self.output_parquet)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write parquet
        self.logger.info(f"Writing parquet to {self.output_parquet}")
        df.to_parquet(self.output_parquet, index=False, compression='snappy')
        
        # Log file statistics
        file_size = output_path.stat().st_size
        self.logger.info(
            "Parquet export complete",
            path=str(self.output_parquet),
            file_size_mb=f"{file_size / 1024 / 1024:.2f}",
            row_count=len(df),
            schema="thread_id, customer_text, brand_reply, resolution_tier, is_boilerplate, turn_count, timestamp, dropped_branch_count"
        )
        
        # Track metrics
        self.metrics.set("parquet_rows", len(df))
        self.metrics.set("parquet_file_size_mb", file_size / 1024 / 1024)
        
        return str(self.output_parquet)
    
    def generate_decision_log(self) -> str:
        """
        Generate decision_log.md documenting all data cleaning decisions.
        
        Args:
            None (uses metrics from self.metrics)
        
        Returns:
            Path to generated decision log
        """
        self.logger.debug("Generating decision log")
        
        metrics = self.metrics.to_dict()
        
        # Build markdown content
        content = []
        content.append("# Phase 1 Data Cleaning Decision Log\n")
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset Overview
        content.append("## Dataset Overview\n")
        content.append(f"- Uber Support Handle(s): {', '.join(self.config.get('UBER_HANDLES', ['Uber_Support']))}\n")
        content.append(f"- Threads Found: {metrics.get('threads_reconstructed', 'N/A')}\n")
        content.append(f"- Threads After Deduplication: {metrics.get('parquet_rows', 'N/A')}\n")
        content.append("")
        
        # Major Processing Steps
        content.append("## Major Processing Steps\n")
        
        # Thread Reconstruction
        content.append("### Thread Reconstruction\n")
        content.append("- Branch Selection Rule: When a tweet has multiple valid replies, the chronologically earliest (lowest tweet_id) is kept; others are discarded to avoid parallel resolution paths\n")
        content.append(f"- Threads with Branching Decisions: {metrics.get('threads_with_branches', 'N/A')}\n")
        content.append(f"- Total Dropped Branches: {metrics.get('dropped_branches', 0)}\n")
        content.append("")
        
        # Resolution Classification
        content.append("### Resolution Classification\n")
        content.append(f"- Resolved Explicit (customer confirmed): {metrics.get('resolution_explicit_count', 'N/A')}\n")
        content.append(f"- Resolved Implicit (brand last, no customer response): {metrics.get('resolution_implicit_count', 'N/A')}\n")
        content.append(f"- Unresolved/Ongoing (ambiguous): {metrics.get('resolution_unresolved_count', 'N/A')}\n")
        content.append("- Note: Implicit resolution has medium confidence; silence does not guarantee satisfaction\n")
        content.append("")
        
        # Boilerplate Detection
        content.append("### Boilerplate Detection\n")
        content.append(f"- Min Occurrences Threshold: {self.config.get('BOILERPLATE_MIN_OCCURRENCES', 5)}\n")
        content.append(f"- Min Share Threshold: {100 * self.config.get('BOILERPLATE_MIN_SHARE', 0.005):.1f}%\n")
        content.append(f"- Unique Boilerplate Signatures: {metrics.get('boilerplate_signatures', 0)}\n")
        content.append(f"- Total Boilerplate Replies: {metrics.get('boilerplate_replies', 0)}\n")
        content.append(f"- Boilerplate Rate: {metrics.get('boilerplate_rate', 0):.2%}\n")
        content.append("")
        
        # Text Normalization & Deduplication
        content.append("### Text Normalization & Deduplication\n")
        content.append(f"- Language Filtering: {self.config.get('LANGUAGE_FILTERING', True)}\n")
        content.append(f"- Accepted Languages: {', '.join(self.config.get('ACCEPTED_LANGUAGES', ['en']))}\n")
        content.append(f"- Threads Removed by Language Filter: {metrics.get('language_filtered_threads', 0)}\n")
        content.append(f"- Empty Text Removed: {metrics.get('dedup_empty_removed', 0)}\n")
        content.append(f"- Duplicates Removed: {metrics.get('dedup_duplicates_removed', 0)}\n")
        content.append("")
        
        # Known Limitations
        content.append("## Known Limitations\n")
        content.append("- **Resolved_implicit threads may include unresolved issues**: Where customer simply stopped replying without indicating satisfaction\n")
        content.append("- **Branch-selection rule discards parallel resolution paths**: May lose valid alternative resolutions that could have been more helpful\n")
        content.append("- **Boilerplate detection relies on text normalization**: Semantically similar but syntactically different replies may not be flagged as boilerplate\n")
        content.append("- **Language detection may misclassify code snippets, URLs, or mixed-language content**: Affecting filtering accuracy\n")
        content.append("")
        
        # Metrics Summary
        content.append("## Metrics Summary\n")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                content.append(f"- {key}: {value:.2f}\n")
            else:
                content.append(f"- {key}: {value}\n")
        
        # Write file
        log_path = Path(self.decision_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(content))
            
            self.logger.info(
                "Decision log generated",
                path=str(self.decision_log_path)
            )
        except Exception as e:
            self.logger.warning(
                "Failed to generate decision log",
                error=str(e)
            )
        
        return str(self.decision_log_path)