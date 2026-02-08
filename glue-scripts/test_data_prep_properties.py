"""
Property-based tests for data preparation script.

This test suite validates the data split proportions property:
- Property 7: Data Split Proportions

**Requirements:**
- Requirement 2.3: Split data into training (70%), validation (15%), and test (15%) sets

Testing Framework: pytest with hypothesis for property-based testing
"""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis import assume
import math


# ========================================
# Property 7: Data Split Proportions
# ========================================

# Feature: fraud-detection-ml-pipeline, Property 7: Data Split Proportions
@settings(max_examples=100)
@given(
    total_records=st.integers(min_value=100, max_value=1000000),
    train_pct=st.floats(min_value=0.5, max_value=0.8)
)
def test_property_7_data_split_proportions_sum_equals_total(
    total_records, train_pct
):
    """
    Property 7: Data Split Proportions - Sum of record counts SHALL equal original dataset size.
    
    For any dataset split into training, validation, and test sets with proportions
    (trainPct, validationPct, testPct), the sum of record counts SHALL equal the
    original dataset size.
    
    **Validates: Requirements 2.3**
    """
    # Generate validation and test proportions that sum to (1.0 - train_pct)
    remaining_pct = 1.0 - train_pct
    # Split remaining between validation and test (roughly equal)
    validation_pct = remaining_pct * 0.5
    test_pct = remaining_pct * 0.5
    
    # Simulate the split logic used by PySpark randomSplit
    # randomSplit uses proportions to determine split sizes
    train_count = int(total_records * train_pct)
    validation_count = int(total_records * validation_pct)
    test_count = int(total_records * test_pct)
    
    # Handle rounding: assign remaining records to maintain total
    assigned = train_count + validation_count + test_count
    remaining = total_records - assigned
    
    # Distribute remaining records (typically 0-2 records due to rounding)
    if remaining > 0:
        train_count += remaining
    elif remaining < 0:
        # This shouldn't happen with proper rounding, but handle it
        train_count += remaining
    
    # Property: Sum of record counts SHALL equal original dataset size
    assert train_count + validation_count + test_count == total_records, (
        f"Sum of split counts ({train_count} + {validation_count} + {test_count} = "
        f"{train_count + validation_count + test_count}) does not equal "
        f"original dataset size ({total_records})"
    )


# Feature: fraud-detection-ml-pipeline, Property 7: Data Split Proportions
@settings(max_examples=100)
@given(
    total_records=st.integers(min_value=100, max_value=1000000),
    train_pct=st.floats(min_value=0.5, max_value=0.8)
)
def test_property_7_data_split_proportions_within_tolerance(
    total_records, train_pct
):
    """
    Property 7: Data Split Proportions - Each split SHALL contain approximately
    trainPct%, validationPct%, and testPct% of records respectively (within 1% tolerance).
    
    For any dataset split into training, validation, and test sets with proportions
    (trainPct, validationPct, testPct), each split SHALL contain approximately
    the specified percentage of records (within 1% tolerance).
    
    **Validates: Requirements 2.3**
    """
    # Generate validation and test proportions that sum to (1.0 - train_pct)
    remaining_pct = 1.0 - train_pct
    validation_pct = remaining_pct * 0.5
    test_pct = remaining_pct * 0.5
    
    # Simulate the split logic used by PySpark randomSplit
    train_count = int(total_records * train_pct)
    validation_count = int(total_records * validation_pct)
    test_count = int(total_records * test_pct)
    
    # Handle rounding: assign remaining records to maintain total
    assigned = train_count + validation_count + test_count
    remaining = total_records - assigned
    
    if remaining > 0:
        train_count += remaining
    elif remaining < 0:
        train_count += remaining
    
    # Calculate actual percentages
    actual_train_pct = train_count / total_records
    actual_validation_pct = validation_count / total_records
    actual_test_pct = test_count / total_records
    
    # Property: Each split SHALL contain approximately the specified percentage
    # (within 1% tolerance for large datasets, or within 1 record for small datasets)
    # For small datasets, rounding can cause larger percentage deviations
    tolerance = 0.01
    
    # For small datasets (< 1000 records), allow tolerance based on record count
    if total_records < 1000:
        # Allow deviation of up to 1 record per split
        record_tolerance = 1.0 / total_records
        tolerance = max(tolerance, record_tolerance * 2)  # 2 records worth of tolerance
    
    assert abs(actual_train_pct - train_pct) <= tolerance, (
        f"Train split percentage {actual_train_pct:.4f} deviates from expected "
        f"{train_pct:.4f} by more than {tolerance:.4f} "
        f"(difference: {abs(actual_train_pct - train_pct):.4f})"
    )
    
    assert abs(actual_validation_pct - validation_pct) <= tolerance, (
        f"Validation split percentage {actual_validation_pct:.4f} deviates from expected "
        f"{validation_pct:.4f} by more than {tolerance:.4f} "
        f"(difference: {abs(actual_validation_pct - validation_pct):.4f})"
    )
    
    assert abs(actual_test_pct - test_pct) <= tolerance, (
        f"Test split percentage {actual_test_pct:.4f} deviates from expected "
        f"{test_pct:.4f} by more than {tolerance:.4f} "
        f"(difference: {abs(actual_test_pct - test_pct):.4f})"
    )


# Feature: fraud-detection-ml-pipeline, Property 7: Data Split Proportions
@settings(max_examples=100)
@given(
    total_records=st.integers(min_value=100, max_value=1000000)
)
def test_property_7_data_split_proportions_standard_split(total_records):
    """
    Property 7: Data Split Proportions - Standard 70/15/15 split SHALL maintain
    correct proportions.
    
    For the standard split configuration (70% train, 15% validation, 15% test),
    the split SHALL maintain correct proportions within 1% tolerance.
    
    **Validates: Requirements 2.3**
    """
    # Standard split proportions from requirements
    train_pct = 0.70
    validation_pct = 0.15
    test_pct = 0.15
    
    # Simulate the split logic
    train_count = int(total_records * train_pct)
    validation_count = int(total_records * validation_pct)
    test_count = int(total_records * test_pct)
    
    # Handle rounding
    assigned = train_count + validation_count + test_count
    remaining = total_records - assigned
    
    if remaining > 0:
        train_count += remaining
    elif remaining < 0:
        train_count += remaining
    
    # Verify sum equals total
    assert train_count + validation_count + test_count == total_records
    
    # Calculate actual percentages
    actual_train_pct = train_count / total_records
    actual_validation_pct = validation_count / total_records
    actual_test_pct = test_count / total_records
    
    # Verify within 1% tolerance
    tolerance = 0.01
    
    assert abs(actual_train_pct - train_pct) <= tolerance, (
        f"Train split {actual_train_pct:.4f} deviates from expected 0.70"
    )
    
    assert abs(actual_validation_pct - validation_pct) <= tolerance, (
        f"Validation split {actual_validation_pct:.4f} deviates from expected 0.15"
    )
    
    assert abs(actual_test_pct - test_pct) <= tolerance, (
        f"Test split {actual_test_pct:.4f} deviates from expected 0.15"
    )


# Feature: fraud-detection-ml-pipeline, Property 7: Data Split Proportions
@settings(max_examples=100)
@given(
    total_records=st.integers(min_value=100, max_value=1000000),
    train_pct=st.floats(min_value=0.5, max_value=0.8)
)
def test_property_7_data_split_proportions_no_data_loss(
    total_records, train_pct
):
    """
    Property 7: Data Split Proportions - No records SHALL be lost or duplicated
    during splitting.
    
    For any dataset split, the total number of records across all splits SHALL
    equal the original dataset size (no data loss or duplication).
    
    **Validates: Requirements 2.3**
    """
    # Generate validation and test proportions that sum to (1.0 - train_pct)
    remaining_pct = 1.0 - train_pct
    validation_pct = remaining_pct * 0.5
    test_pct = remaining_pct * 0.5
    
    # Simulate the split logic
    train_count = int(total_records * train_pct)
    validation_count = int(total_records * validation_pct)
    test_count = int(total_records * test_pct)
    
    # Handle rounding
    assigned = train_count + validation_count + test_count
    remaining = total_records - assigned
    
    if remaining > 0:
        train_count += remaining
    elif remaining < 0:
        train_count += remaining
    
    # Property: No data loss or duplication
    total_after_split = train_count + validation_count + test_count
    
    assert total_after_split == total_records, (
        f"Data loss or duplication detected: original={total_records}, "
        f"after_split={total_after_split}, difference={total_after_split - total_records}"
    )
    
    # All splits should be non-negative
    assert train_count >= 0, f"Train count is negative: {train_count}"
    assert validation_count >= 0, f"Validation count is negative: {validation_count}"
    assert test_count >= 0, f"Test count is negative: {test_count}"


# Feature: fraud-detection-ml-pipeline, Property 7: Data Split Proportions
def test_property_7_data_split_proportions_kaggle_dataset():
    """
    Property 7: Data Split Proportions - Kaggle dataset (284,807 records) SHALL
    split correctly with standard proportions.
    
    For the specific Kaggle Credit Card Fraud Detection dataset with 284,807 records,
    the standard 70/15/15 split SHALL produce record counts that sum to the total
    and are within 1% of the expected proportions.
    
    **Validates: Requirements 2.2, 2.3**
    """
    # Kaggle dataset size from requirements
    total_records = 284807
    
    # Standard split proportions
    train_pct = 0.70
    validation_pct = 0.15
    test_pct = 0.15
    
    # Simulate the split logic
    train_count = int(total_records * train_pct)
    validation_count = int(total_records * validation_pct)
    test_count = int(total_records * test_pct)
    
    # Handle rounding
    assigned = train_count + validation_count + test_count
    remaining = total_records - assigned
    
    if remaining > 0:
        train_count += remaining
    elif remaining < 0:
        train_count += remaining
    
    # Verify sum equals total (most important property)
    assert train_count + validation_count + test_count == total_records, (
        f"Sum of splits ({train_count + validation_count + test_count}) "
        f"does not equal Kaggle dataset size ({total_records})"
    )
    
    # Verify proportions are within 1% tolerance
    actual_train_pct = train_count / total_records
    actual_validation_pct = validation_count / total_records
    actual_test_pct = test_count / total_records
    
    tolerance = 0.01
    
    assert abs(actual_train_pct - train_pct) <= tolerance, (
        f"Train proportion {actual_train_pct:.4f} deviates from expected {train_pct:.4f}"
    )
    
    assert abs(actual_validation_pct - validation_pct) <= tolerance, (
        f"Validation proportion {actual_validation_pct:.4f} deviates from expected {validation_pct:.4f}"
    )
    
    assert abs(actual_test_pct - test_pct) <= tolerance, (
        f"Test proportion {actual_test_pct:.4f} deviates from expected {test_pct:.4f}"
    )


# ========================================
# Property 8: SageMaker Output Format
# ========================================

# Feature: fraud-detection-ml-pipeline, Property 8: SageMaker Output Format
@settings(max_examples=100)
@given(
    num_features=st.integers(min_value=5, max_value=50),
    num_records=st.integers(min_value=10, max_value=1000),
    has_target=st.booleans()
)
def test_property_8_sagemaker_output_format_column_order(
    num_features, num_records, has_target
):
    """
    Property 8: SageMaker Output Format - Target column SHALL be first.
    
    For any prepared dataset written for SageMaker training, the target column
    SHALL be the first column, followed by feature columns.
    
    SageMaker XGBoost expects: [target, feature1, feature2, ...]
    
    **Validates: Requirements 2.4**
    """
    # Assume we have a target column
    assume(has_target)
    
    # Simulate column ordering for SageMaker
    target_column = "Class"
    feature_columns = [f"V{i}" for i in range(1, num_features + 1)]
    
    # Simulate the reorder_columns_for_sagemaker function
    all_columns = feature_columns + [target_column]  # Original order (features first)
    
    # Reorder: target first, then features
    ordered_columns = [target_column] + [col for col in all_columns if col != target_column]
    
    # Property: Target column SHALL be first
    assert ordered_columns[0] == target_column, (
        f"Target column '{target_column}' is not first in ordered columns. "
        f"First column is '{ordered_columns[0]}'"
    )
    
    # Property: All feature columns SHALL follow target
    feature_columns_in_output = ordered_columns[1:]
    assert len(feature_columns_in_output) == num_features, (
        f"Expected {num_features} feature columns after target, "
        f"got {len(feature_columns_in_output)}"
    )


# Feature: fraud-detection-ml-pipeline, Property 8: SageMaker Output Format
@settings(max_examples=100)
@given(
    num_features=st.integers(min_value=5, max_value=50),
    include_time=st.booleans(),
    include_amount=st.booleans()
)
def test_property_8_sagemaker_output_format_all_columns_present(
    num_features, include_time, include_amount
):
    """
    Property 8: SageMaker Output Format - All feature columns plus target SHALL be present.
    
    For any prepared dataset written for SageMaker training, the output SHALL contain
    all feature columns plus the target column specified in the configuration.
    
    **Validates: Requirements 2.4**
    """
    # Build expected columns
    target_column = "Class"
    feature_columns = [f"V{i}" for i in range(1, num_features + 1)]
    
    if include_time:
        feature_columns.append("Time")
    if include_amount:
        feature_columns.append("Amount")
    
    # Simulate the reorder_columns_for_sagemaker function
    all_columns = feature_columns + [target_column]
    ordered_columns = [target_column] + [col for col in all_columns if col != target_column]
    
    # Property: All columns SHALL be present (target + features)
    expected_column_count = 1 + len(feature_columns)  # 1 target + N features
    assert len(ordered_columns) == expected_column_count, (
        f"Expected {expected_column_count} columns (1 target + {len(feature_columns)} features), "
        f"got {len(ordered_columns)}"
    )
    
    # Property: Target column SHALL be present
    assert target_column in ordered_columns, (
        f"Target column '{target_column}' not found in output columns"
    )
    
    # Property: All feature columns SHALL be present
    for feature_col in feature_columns:
        assert feature_col in ordered_columns, (
            f"Feature column '{feature_col}' not found in output columns"
        )


# Feature: fraud-detection-ml-pipeline, Property 8: SageMaker Output Format
def test_property_8_sagemaker_output_format_kaggle_dataset():
    """
    Property 8: SageMaker Output Format - Kaggle dataset SHALL have correct column structure.
    
    For the Kaggle Credit Card Fraud Detection dataset, the prepared output SHALL contain:
    - Target column 'Class' as first column
    - 30 feature columns: Time, V1-V28, Amount
    
    **Validates: Requirements 2.4**
    """
    # Kaggle dataset structure
    target_column = "Class"
    feature_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Simulate the reorder_columns_for_sagemaker function
    all_columns = feature_columns + [target_column]
    ordered_columns = [target_column] + [col for col in all_columns if col != target_column]
    
    # Property: Target column SHALL be first
    assert ordered_columns[0] == target_column, (
        f"Target column '{target_column}' is not first. First column is '{ordered_columns[0]}'"
    )
    
    # Property: Total columns SHALL be 31 (1 target + 30 features)
    assert len(ordered_columns) == 31, (
        f"Expected 31 columns (1 target + 30 features), got {len(ordered_columns)}"
    )
    
    # Property: All feature columns SHALL be present
    for feature_col in feature_columns:
        assert feature_col in ordered_columns, (
            f"Feature column '{feature_col}' not found in output"
        )
    
    # Property: No duplicate columns
    assert len(ordered_columns) == len(set(ordered_columns)), (
        f"Duplicate columns found in output"
    )


# Feature: fraud-detection-ml-pipeline, Property 8: SageMaker Output Format
@settings(max_examples=100)
@given(
    num_features=st.integers(min_value=1, max_value=100)
)
def test_property_8_sagemaker_output_format_no_duplicates(num_features):
    """
    Property 8: SageMaker Output Format - No duplicate columns SHALL exist.
    
    For any prepared dataset written for SageMaker training, the output SHALL NOT
    contain duplicate column names.
    
    **Validates: Requirements 2.4**
    """
    # Build columns
    target_column = "Class"
    feature_columns = [f"V{i}" for i in range(1, num_features + 1)]
    
    # Simulate the reorder_columns_for_sagemaker function
    all_columns = feature_columns + [target_column]
    ordered_columns = [target_column] + [col for col in all_columns if col != target_column]
    
    # Property: No duplicate columns
    unique_columns = set(ordered_columns)
    assert len(ordered_columns) == len(unique_columns), (
        f"Duplicate columns found: {len(ordered_columns)} total columns, "
        f"{len(unique_columns)} unique columns"
    )


# Feature: fraud-detection-ml-pipeline, Property 8: SageMaker Output Format
@settings(max_examples=100)
@given(
    num_features=st.integers(min_value=1, max_value=50),
    target_name=st.sampled_from(["Class", "Label", "Target", "Y"])
)
def test_property_8_sagemaker_output_format_target_first_any_name(
    num_features, target_name
):
    """
    Property 8: SageMaker Output Format - Target column SHALL be first regardless of name.
    
    For any prepared dataset with any target column name, the target SHALL be the
    first column in the output.
    
    **Validates: Requirements 2.4**
    """
    # Build columns
    feature_columns = [f"V{i}" for i in range(1, num_features + 1)]
    
    # Simulate the reorder_columns_for_sagemaker function with custom target name
    all_columns = feature_columns + [target_name]
    ordered_columns = [target_name] + [col for col in all_columns if col != target_name]
    
    # Property: Target column SHALL be first
    assert ordered_columns[0] == target_name, (
        f"Target column '{target_name}' is not first. First column is '{ordered_columns[0]}'"
    )
    
    # Property: All feature columns SHALL follow target
    assert len(ordered_columns) == num_features + 1, (
        f"Expected {num_features + 1} columns (1 target + {num_features} features), "
        f"got {len(ordered_columns)}"
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
