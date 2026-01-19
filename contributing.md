# Contributing to PASTO

Thank you for your interest in contributing to PASTO! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU info)
- Error messages and stack traces

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternative solutions you've considered

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/pasto.git
   cd pasto
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, commented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black src/
   flake8 src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Clear description of changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Add comments for complex logic

Example:
```python
def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute evaluation metric
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Metric value
    """
    # Implementation
    pass
```

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

```python
def test_new_feature():
    """Test description"""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = new_feature(input_data)
    
    # Assert
    assert result.shape == expected_shape
    assert result.min() >= 0
```

### Documentation

- Update README.md if adding new features
- Add docstrings to all new functions/classes
- Update configuration documentation
- Include usage examples

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pasto.git
cd pasto

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt
```

## Project Structure

```
pasto/
├── src/
│   ├── models/          # Model implementations
│   ├── data/            # Data loading and processing
│   ├── training/        # Training utilities
│   ├── evaluation/      # Metrics and evaluation
│   └── utils/           # Helper utilities
├── baselines/           # Baseline implementations
├── configs/             # Configuration files
├── scripts/             # Utility scripts
├── tests/               # Unit tests
└── notebooks/           # Jupyter notebooks
```

## Areas for Contribution

### High Priority
- [ ] Additional baseline implementations
- [ ] More efficient RMAB algorithms
- [ ] Support for additional datasets
- [ ] Improved documentation and tutorials
- [ ] Performance optimizations

### Medium Priority
- [ ] Visualization improvements
- [ ] Additional fairness metrics
- [ ] Hyperparameter tuning utilities
- [ ] Distributed training support

### Nice to Have
- [ ] Web dashboard for results
- [ ] Interactive policy visualization
- [ ] Automated experiment tracking
- [ ] Docker containerization

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion on GitHub Discussions
- Email the maintainers

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Academic papers (for significant contributions)

Thank you for contributing to PASTO!
