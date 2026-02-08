# Glue Scripts Property Tests

This directory contains property-based tests for the AWS Glue data preparation scripts.

## Test Files

- `test_data_prep_properties.py` - Property tests for data split proportions (Property 7)

## Running Tests

### Install Dependencies

```bash
pip install -r test-requirements.txt
```

### Run All Tests

```bash
pytest test_data_prep_properties.py -v
```

### Run Specific Test

```bash
pytest test_data_prep_properties.py::test_property_7_data_split_proportions_sum_equals_total -v
```

## Property 7: Data Split Proportions

This property validates that dataset splitting maintains correct proportions:

1. **Sum Equals Total**: The sum of train, validation, and test record counts equals the original dataset size
2. **Within Tolerance**: Each split contains approximately the specified percentage (within 1% for large datasets)
3. **Standard Split**: The standard 70/15/15 split maintains correct proportions
4. **No Data Loss**: No records are lost or duplicated during splitting
5. **Kaggle Dataset**: The specific Kaggle dataset (284,807 records) splits correctly

## Testing Framework

- **pytest**: Test runner
- **hypothesis**: Property-based testing library (Python equivalent of Kotest)
- **Settings**: 100 examples per property test for thorough validation

## Notes

- For small datasets (< 1000 records), the tolerance is adjusted to account for rounding effects
- The tests simulate the split logic used by PySpark's `randomSplit` method
- All tests validate Requirement 2.3 from the design document
