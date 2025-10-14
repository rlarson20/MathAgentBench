# Contributing

## Development Setup

```bash
# Clone and install
git clone https://github.com/yourusername/mathagent-bench.git
cd mathagent-bench
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=mathagent_bench --cov-report=html

# Specific test file
pytest tests/test_evaluator.py -v
```

## Code Style

We use:

- **ruff** for formatting and linting
- **mypy** for type checking

```bash

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Pull Request Process

1. **Fork** the repo
2. **Create branch**: `git checkout -b feature/your-feature`
3. **Make changes** with tests
4. **Commit**: Use clear commit messages, preferably conventional commits
5. **Push** and open PR

## Testing Guidelines

- Write tests for new features
- Aim for >80% coverage
- Use fixtures from `tests/conftest.py`
- Mock external APIs (OpenRouter, Docker)

## Documentation

- Update README.md for user-facing changes
- Update docs/architecture.md for design changes
- Docstrings for public APIs

## Adding Benchmarks

To contribute a new benchmark:

1. Follow schema in `benchmarks/README.md`
2. Include diverse difficulty levels
3. Verify answers are correct
4. Add metadata (tags, difficulty)

## Questions?

Open an issue or discussion on GitHub.
