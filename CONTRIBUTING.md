# Contributing to ATLAS

Thank you for your interest in contributing to ATLAS! We value all contributions, whether it's code, documentation, bug reports, feature requests, or community support.

## Ways to Contribute

- **Fix bugs**: Address issues in existing code
- **Add features**: Implement new capabilities or trainers
- **Improve documentation**: Enhance clarity, fix typos, add examples
- **Report issues**: Help us identify bugs and areas for improvement
- **Answer questions**: Support the community in discussions

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone git@github.com:<your-username>/ATLAS.git
cd ATLAS
git remote add upstream https://github.com/Arc-Computer/ATLAS.git
```

2. **Create a development environment**:
```bash
conda create -n atlas-dev python=3.11
conda activate atlas-dev
pip install -e .[dev]
```

3. **Install pre-commit hooks**:
```bash
pre-commit install
```

### Development Workflow

1. **Sync with upstream**:
```bash
git checkout main
git fetch upstream
git merge upstream/main
```

2. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes** and ensure tests pass:
```bash
pytest tests/
make lint  # Run code quality checks
```

4. **Commit with descriptive messages**:
```bash
git add .
git commit -m "feat: add new teacher training algorithm"
```

5. **Push and create a pull request**:
```bash
git push -u origin feature/your-feature-name
```

## Submitting Issues

### Bug Reports

Include the following information:
- Python version
- PyTorch version
- ATLAS version (`pip show atlas`)
- CUDA version and GPU model
- Minimal reproducible code example
- Full error traceback

Run `atlas env` to automatically collect environment information.

### Feature Requests

Describe:
1. **Motivation**: Problem or use case driving the request
2. **Proposed solution**: Detailed description with examples
3. **Alternatives considered**: Other approaches you've evaluated
4. **Related work**: Links to papers or implementations

## Code Standards

### Style Guide

We use `ruff` for code formatting and linting:
```bash
make format  # Auto-format code
make lint    # Check code quality
```

### Documentation

All public functions must include:
```python
def train_teacher(
    model: nn.Module,
    config: TrainerConfig,
    dataset: Dataset,
) -> nn.Module:
    """
    Train a teacher model using GRPO.

    Args:
        model (`nn.Module`):
            Base model to train.
        config (`TrainerConfig`):
            Training configuration.
        dataset (`Dataset`):
            Training dataset.

    Returns:
        `nn.Module`: Trained teacher model.

    Raises:
        ValueError: If config parameters are invalid.
        RuntimeError: If training fails.

    Examples:
        ```python
        >>> model = AutoModelForCausalLM.from_pretrained("llama-3.1-8b")
        >>> config = TrainerConfig(learning_rate=1e-5)
        >>> trained = train_teacher(model, config, dataset)
        ```
    """
```

### Testing

Add tests for new features:
```python
def test_new_feature():
    """Test description."""
    # Arrange
    model = create_test_model()

    # Act
    result = new_feature(model)

    # Assert
    assert result.shape == expected_shape
    assert torch.allclose(result, expected_output)
```

Run tests with coverage:
```bash
pytest tests/ --cov=atlas --cov-report=html
```

## Pull Request Checklist

Before submitting:

- [ ] Code follows style guidelines (`make lint` passes)
- [ ] Tests added/updated and passing
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format
- [ ] PR title summarizes the contribution
- [ ] Linked to relevant issue (fixes #123)

## Implementing New Trainers

When adding a new training algorithm:

1. **Open an issue first** with:
   - Paper link and summary
   - Implementation complexity
   - Performance benchmarks
   - Use cases

2. **Follow the trainer template**:
   - Inherit from `BaseTrainer`
   - Implement required methods
   - Add configuration class
   - Include comprehensive tests

3. **Document thoroughly**:
   - Add to trainer documentation
   - Include usage examples
   - Provide benchmark results

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints

### Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests

### Response Times

We aim to:
- Triage new issues within 48 hours
- Review PRs within 1 week
- Merge approved PRs within 2 weeks

## Recognition

Contributors are recognized in:
- Release notes
- Contributors file
- Annual community report

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Questions?

Feel free to ask in:
- [GitHub Discussions](https://github.com/Arc-Computer/ATLAS/discussions)
- [GitHub Issues](https://github.com/Arc-Computer/ATLAS/issues)

Thank you for contributing to ATLAS! 🚀