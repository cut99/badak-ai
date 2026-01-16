# BADAK AI Worker - Testing Guide

This directory contains comprehensive tests for the BADAK AI Worker.

## Test Structure

```
tests/
├── test_models.py       # AI model tests (InsightFace, OpenCLIP, BLIP)
├── test_clustering.py   # Clustering and VectorDB tests
└── test_api.py          # API endpoint and security tests
```

## Running Tests

### Prerequisites

Make sure you have installed all dependencies including test dependencies:

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# From python-worker directory
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Test AI models only
pytest tests/test_models.py -v

# Test clustering service only
pytest tests/test_clustering.py -v

# Test API endpoints only
pytest tests/test_api.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_models.py::TestInsightFaceModel -v

# Run specific test function
pytest tests/test_models.py::TestInsightFaceModel::test_model_initialization -v
```

### Run Tests by Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

### Test Models (`test_models.py`)

Tests for AI models with CPU-based testing:

- **InsightFaceModel**
  - Model initialization with CPU provider
  - Face detection on sample images
  - Face cropping functionality
  - Model information retrieval
  - Face object structure validation

- **OpenCLIPModel**
  - Model initialization and configuration
  - Zero-shot tag classification
  - Custom threshold handling
  - Tag scores retrieval
  - Predefined tags validation

- **BLIPModel**
  - Model initialization
  - Caption generation
  - Indonesian context phrase mapping
  - English to Indonesian keyword matching
  - Fallback behavior testing

- **Integration Tests**
  - Full pipeline with all three models
  - End-to-end processing verification

### Test Clustering (`test_clustering.py`)

Tests for clustering logic and VectorDB operations:

- **ClusteringService**
  - New cluster creation
  - Similar face detection and assignment
  - Different face separation
  - Cluster merging with thumbnail cleanup
  - Cluster statistics and info retrieval
  - Similarity threshold updates

- **VectorDBService**
  - Database initialization with ChromaDB
  - Face embedding storage
  - Similarity search with cosine distance
  - Cluster assignment updates
  - Face retrieval by cluster
  - Database information queries

### Test API (`test_api.py`)

Tests for REST API endpoints and security:

- **Health Endpoint**
  - Health check response structure
  - No authentication requirement
  - Model loading status

- **Process Endpoint**
  - API key requirement
  - Request validation
  - Missing field handling
  - Invalid URL handling

- **Merge Clusters Endpoint**
  - API key requirement
  - Source/target validation
  - Empty list handling
  - Self-merge prevention

- **Thumbnail Endpoint**
  - API key requirement
  - 404 handling for missing thumbnails
  - Content-Type validation

- **Security Middleware**
  - API key validation
  - Invalid key blocking
  - Exempt paths (health, docs)
  - IP whitelist functionality

- **CORS Middleware**
  - CORS headers presence
  - Cross-origin request handling

## Test Configuration

Tests are configured via `pytest.ini`:

- **Test Discovery**: Automatically finds `test_*.py` files
- **Output**: Verbose mode with colored output
- **Coverage**: Source code coverage tracking
- **Async Support**: Auto-mode for async/await tests

## Writing New Tests

### Test Fixtures

Use existing fixtures for common test setup:

```python
@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')

def test_my_function(sample_image):
    # Use the fixture
    result = process_image(sample_image)
    assert result is not None
```

### Temporary Directories

For tests requiring file I/O:

```python
@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)
```

### Async Tests

For testing async functions:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await download_image("http://example.com/image.jpg")
    assert result is not None
```

## Notes

- **CPU Testing**: All tests run on CPU to ensure compatibility
- **Temporary Data**: Tests use temporary directories for VectorDB and thumbnails
- **No External Dependencies**: Tests use mock data and don't require external services
- **Clean State**: Each test has isolated state via fixtures

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=. --cov-report=xml
```

## Troubleshooting

### Model Download Issues

If models fail to download during tests:
- Ensure you have internet connectivity
- Models are cached after first download
- Check proxy settings if behind firewall

### ChromaDB Errors

If VectorDB tests fail:
- Ensure write permissions in temp directory
- Check available disk space
- Verify ChromaDB version matches requirements.txt

### Memory Issues

If tests run out of memory:
- Run tests individually instead of all at once
- Use `--maxfail=1` to stop after first failure
- Close other applications to free memory

## Coverage Reports

Generate HTML coverage report:

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # Mac/Linux
start htmlcov/index.html # Windows
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass before committing
3. Maintain >80% code coverage
4. Add docstrings to test functions
