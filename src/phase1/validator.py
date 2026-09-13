"""Validator: Quality assurance for Phase 1 output."""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from .models import ParquetSchema
from .logger import StructuredLogger


class ValidationReport:
    """Report on Phase 1 output validation."""
    
    def __init__(self):
        """Initialize validation report."""
        self.status = "PENDING"
        self.row_count_check = {}
        self.schema_check = {}
        self.sample_review = {}
        self.statistics = {}
        self.data_quality_issues = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "status": self.status,
            "row_count_check": self.row_count_check,
            "schema_check": self.schema_check,
            "sample_review": self.sample_review,
            "statistics": self.statistics,
            "data_quality_issues": self.data_quality_issues
        }


class Validator:
    """Validate Phase 1 output against acceptance criteria."""
    
    def __init__(self, config: Dict[str, Any], logger: StructuredLogger):
        """
        Initialize Validator.
        
        Args:
            config: Configuration dictionary
            logger: StructuredLogger instance
        """
        self.config = config
        self.logger = logger
        self.phase0_projections = config.get("PHASE0_PROJECTIONS", {})
    
    def validate_output(self, df: pd.DataFrame, sample_size: int = 20) -> ValidationReport:
        """
        Validate Phase 1 output against acceptance criteria.
        
        Validation Checks:
        1. Row count comparison vs Phase 0 projections (±50%)
        2. Schema validation
        3. Sample review (20 threads)
        4. Data quality checks
        5. Statistics plausibility
        
        Args:
            df: Output DataFrame from parquet
            sample_size: Number of threads to sample (default 20)
        
        Returns:
            ValidationReport with detailed results
        """
        report = ValidationReport()
        
        self.logger.info("Starting Phase 1 output validation")
        
        # Check 1: Row Count Validation
        self.logger.debug("Check 1: Row count validation")
        report.row_count_check = self._validate_row_count(df)
        if report.row_count_check["result"] == "FAIL":
            report.status = "FAIL"
        elif report.row_count_check["result"] == "WARNING":
            report.status = "WARNING"
        else:
            report.status = "PASS"
        
        # Check 2: Schema Validation
        self.logger.debug("Check 2: Schema validation")
        report.schema_check = self._validate_schema(df)
        if report.schema_check["result"] == "FAIL":
            report.status = "FAIL"
        
        # Check 3: Sample Review
        self.logger.debug("Check 3: Sample review")
        report.sample_review = self._sample_review(df, sample_size)
        
        # Check 4: Data Quality
        self.logger.debug("Check 4: Data quality checks")
        issues = self._check_data_quality(df)
        report.data_quality_issues = issues
        if issues:
            report.status = "FAIL"
        
        # Check 5: Statistics
        self.logger.debug("Check 5: Statistics plausibility")
        report.statistics = self._calculate_statistics(df)
        
        # Final verdict
        self.logger.info(
            "Validation complete",
            status=report.status,
            row_count_result=report.row_count_check.get("result"),
            schema_result=report.schema_check.get("result"),
            data_quality_issues=len(issues)
        )
        
        return report
    
    def _validate_row_count(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate row count vs Phase 0 projections."""
        expected = self.phase0_projections.get("thread_count")
        actual = len(df)
        
        if expected is None:
            return {"result": "PASS", "message": "No Phase 0 projection available"}
        
        deviation_pct = 100 * abs(actual - expected) / expected if expected > 0 else 0
        
        if deviation_pct > 100:
            result = "FAIL"
        elif deviation_pct > 50:
            result = "WARNING"
        else:
            result = "PASS"
        
        return {
            "expected": expected,
            "actual": actual,
            "deviation_pct": f"{deviation_pct:.1f}%",
            "result": result
        }
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate output schema."""
        try:
            ParquetSchema.validate_dataframe(df)
            return {"result": "PASS", "mismatches": []}
        except TypeError as e:
            return {"result": "FAIL", "mismatches": [str(e)]}
    
    def _sample_review(self, df: pd.DataFrame, sample_size: int = 20) -> Dict[str, Any]:
        """Sample and review threads."""
        if len(df) < sample_size:
            sample_size = len(df)
        
        sampled = df.sample(n=sample_size, random_state=42)
        
        sample_threads = []
        for _, row in sampled.iterrows():
            sample_threads.append({
                "thread_id": row["thread_id"],
                "customer_text": row["customer_text"][:100],  # Truncate for display
                "brand_reply": row["brand_reply"][:100],
                "turn_count": row["turn_count"],
                "resolution_tier": row["resolution_tier"],
                "dropped_branch_count": row["dropped_branch_count"],
            })
        
        self.logger.info(
            f"Sampled {len(sample_threads)} threads for manual review",
            sample_size=len(sample_threads)
        )
        
        return {
            "sampled_count": len(sample_threads),
            "sampled_threads": sample_threads,
            "manual_review_result": "PENDING",
            "reviewer_comments": ""
        }
    
    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check data quality issues."""
        issues = []
        
        # Check for nulls in critical columns
        critical_cols = ["thread_id", "customer_text", "resolution_tier"]
        for col in critical_cols:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"Found {null_count} null values in {col}")
        
        # Check turn_count >= 1
        invalid_turns = (df["turn_count"] < 1).sum()
        if invalid_turns > 0:
            issues.append(f"Found {invalid_turns} threads with turn_count < 1")
        
        # Check resolution_tier values
        valid_tiers = {"resolved_explicit", "resolved_implicit", "unresolved_or_ongoing"}
        invalid_tiers = df[~df["resolution_tier"].isin(valid_tiers)]
        if len(invalid_tiers) > 0:
            issues.append(f"Found {len(invalid_tiers)} threads with invalid resolution_tier")
        
        # Check dropped_branch_count >= 0
        invalid_branches = (df["dropped_branch_count"] < 0).sum()
        if invalid_branches > 0:
            issues.append(f"Found {invalid_branches} threads with dropped_branch_count < 0")
        
        return issues
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate output statistics."""
        stats = {}
        
        # Dropped branch distribution
        stats["dropped_branch_distribution"] = {
            "zero": (df["dropped_branch_count"] == 0).sum(),
            "one": (df["dropped_branch_count"] == 1).sum(),
            "two_plus": (df["dropped_branch_count"] >= 2).sum(),
        }
        
        # Resolution tier distribution
        stats["resolution_tier_distribution"] = df["resolution_tier"].value_counts().to_dict()
        
        # Boilerplate rate
        stats["boilerplate_rate"] = f"{100 * df['is_boilerplate'].sum() / len(df):.2f}%"
        
        # Turn count statistics
        stats["turn_count_mean"] = f"{df['turn_count'].mean():.1f}"
        stats["turn_count_max"] = int(df['turn_count'].max())
        stats["turn_count_min"] = int(df['turn_count'].min())
        
        return stats
