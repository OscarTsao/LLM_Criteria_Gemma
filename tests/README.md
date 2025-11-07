# Tests

Unit and integration tests for the LLM_Criteria_Gemma project.

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_poolers.py -v
pytest tests/test_data.py -v
```

### Run Specific Test Class or Function

```bash
# Run a specific test class
pytest tests/test_poolers.py::TestMeanPooler -v

# Run a specific test function
pytest tests/test_poolers.py::TestMeanPooler::test_mean_pooler_no_mask -v
```

### Run Tests with Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Run Tests in Parallel

```bash
pip install pytest-xdist
pytest tests/ -n auto
```

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Shared fixtures and configuration
├── README.md            # This file
├── test_poolers.py      # Tests for pooling strategies
└── test_data.py         # Tests for data loading and CV splits
```

## Test Markers

Tests can be marked with custom markers:

- `@pytest.mark.slow` - For slow-running tests
- `@pytest.mark.gpu` - For tests requiring GPU
- `@pytest.mark.integration` - For integration tests

### Skip Slow Tests

```bash
pytest tests/ -m "not slow"
```

### Run Only GPU Tests

```bash
pytest tests/ -m "gpu"
```

## Writing New Tests

### Test File Naming

- Test files should start with `test_`
- Test functions should start with `test_`
- Test classes should start with `Test`

### Example Test Structure

```python
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from module import function_to_test

class TestMyFeature:
    """Tests for my feature."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return {"key": "value"}

    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        result = function_to_test(sample_data)
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

## Test Coverage Goals

- **Poolers**: 100% coverage - All pooling strategies tested
- **Data Loading**: 95%+ coverage - Core functionality tested
- **Models**: Integration tests for model initialization
- **Training**: Integration tests for training loop

## CI/CD Integration

These tests are automatically run on:
- Pull requests
- Pushes to main branch
- Nightly builds

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running from the project root:

```bash
cd /path/to/LLM_Criteria_Gemma
pytest tests/
```

### GPU Tests Failing

GPU tests require a CUDA-capable GPU. To skip:

```bash
pytest tests/ -m "not gpu"
```

### Slow Tests

Some tests may be slow due to data processing. Use parallel execution:

```bash
pytest tests/ -n auto
```

## Test Results

After running tests, you'll see output like:

```
tests/test_poolers.py::TestMeanPooler::test_mean_pooler_no_mask PASSED    [ 5%]
tests/test_poolers.py::TestMeanPooler::test_mean_pooler_with_mask PASSED  [10%]
...
======================== 45 passed in 2.34s ========================
```

## Contact

For questions about tests, please open an issue or contact the maintainers.
