# Contributing to LLM_Criteria_Gemma

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LLM_Criteria_Gemma.git
   cd LLM_Criteria_Gemma
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/OscarTsao/LLM_Criteria_Gemma.git
   ```

4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

4. **Verify installation**:
   ```bash
   make check-env
   ```

## Making Changes

### Branching Strategy

- `main` - Stable production code
- `develop` - Integration branch for features
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Development Workflow

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Make your changes** in logical commits

3. **Add tests** for new functionality

4. **Run tests** locally:
   ```bash
   make test
   ```

5. **Format code**:
   ```bash
   make format
   ```

6. **Check code quality**:
   ```bash
   make lint
   ```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_poolers.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests in parallel
pytest tests/ -n auto
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive names that explain what is being tested

Example:
```python
def test_mean_pooler_with_attention_mask():
    """Test that mean pooler correctly handles attention masks."""
    pooler = MeanPooler()
    hidden_states = torch.randn(2, 5, 10)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    result = pooler(hidden_states, attention_mask)

    assert result.shape == (2, 10)
    # Additional assertions...
```

### Test Coverage Goals

- Aim for 80%+ code coverage
- 100% coverage for critical paths (data loading, model forward pass)
- Include edge cases and error conditions

## Code Style

### Python Style Guide

This project follows **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Docstrings**: Google style
- **Type hints**: Required for public APIs
- **Imports**: Organized with `isort`

### Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black src/ tests/

# Check formatting without making changes
black src/ tests/ --check
```

### Linting

We use **flake8** for linting:

```bash
# Run flake8
flake8 src/ tests/

# Configuration in setup.cfg or .flake8
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Run mypy
mypy src/

# Configuration in mypy.ini or setup.cfg
```

### Documentation

- **Docstrings**: All public functions, classes, and methods must have docstrings
- **Format**: Use Google-style docstrings
- **Comments**: Explain *why*, not *what*

Example docstring:
```python
def train_fold(self, model, train_loader, val_loader, class_weights=None):
    """
    Train and evaluate a single fold.

    Args:
        model: GemmaClassifier instance to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_weights: Optional tensor of class weights for loss function

    Returns:
        Tuple of (history dict, best validation F1 score, best epoch number)

    Raises:
        RuntimeError: If training fails due to GPU memory issues
    """
```

## Commit Messages

### Format

Use the following format for commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(poolers): Add weighted average pooling strategy

Implement weighted average pooling that assigns learnable weights
to different token positions. This allows the model to learn which
tokens are most important for the classification task.

Closes #42
```

```
fix(data): Handle missing symptom labels gracefully

Previously crashed when encountering unmapped symptom labels.
Now logs a warning and skips invalid entries.

Fixes #38
```

### Best Practices

- **First line**: Max 72 characters, imperative mood
- **Body**: Explain *what* and *why*, not *how*
- **Footer**: Reference issues/PRs

## Pull Request Process

### Before Submitting

1. ✅ All tests pass locally
2. ✅ Code is formatted with Black
3. ✅ No linting errors
4. ✅ Documentation updated
5. ✅ CHANGELOG.md updated (if applicable)
6. ✅ Branch is up-to-date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Changes Made
- List of specific changes
- Another change

## Testing
- Describe testing performed
- Include test results

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one approval** from maintainers
3. **No merge conflicts** with main branch
4. **All discussions resolved**

### After Approval

- **Squash and merge** is preferred for feature branches
- **Rebase and merge** for clean, linear history
- **Delete branch** after merging

## Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Description**: Clear, concise description
- **Steps to reproduce**: Numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, GPU, etc.
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

### Feature Requests

Use the feature request template and include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Other solutions considered
- **Additional context**: Any other information

### Security Issues

**Do not** open public issues for security vulnerabilities. Instead:

1. Email maintainers directly
2. Provide detailed description
3. Allow time for fix before disclosure

## Development Tips

### Useful Commands

```bash
# Check code quality
make lint

# Format code
make format

# Run tests
make test

# Clean build artifacts
make clean

# View GPU usage
make check-gpu

# Show make targets
make help
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python src/training/train_gemma_hydra.py

# Run with Python debugger
python -m pdb src/training/train_gemma_hydra.py
```

### Performance Profiling

```bash
# Profile training script
python -m cProfile -o profile.stats src/training/train_gemma_hydra.py

# View results
python -m pstats profile.stats
```

## Recognition

Contributors are recognized in:

- **README.md**: Contributors section
- **GitHub**: Contributors graph
- **CHANGELOG.md**: Release notes

## Questions?

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions and ideas
- **Email**: Contact maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to LLM_Criteria_Gemma! 🎉
