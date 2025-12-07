# Contributing

Contributions, improvements, and bug fixes are welcome! Please feel free to submit bug reports, feature requests, and pull requests. If you leverage this project in your own work, please be mindful of the license.

## Development Setup

### Installing Development Dependencies

To set up the development environment, install the package with development dependencies:

```bash
uv sync --extra dev
```

This will install:
- `ruff` - Fast Python linter and formatter
- `pre-commit` - Git hooks for code consistency

### Setting Up Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistent formatting. The hooks run `ruff` for both linting and formatting.

To install the pre-commit hooks:

```bash
pre-commit install
```

After installation, the hooks will automatically run on every commit. The hooks will:
- Run `ruff` linter with auto-fix enabled
- Run `ruff` formatter to ensure consistent code style

You can also manually run the hooks on all files:

```bash
pre-commit run --all-files
```

### Code Style

This project uses `ruff` for both linting and formatting. The configuration is defined in `pyproject.toml`:
- Line length: 120 characters
- Target Python version: 3.10
- Quote style: Double quotes
- Indent style: Spaces

The pre-commit hooks will automatically format your code, but you can also run formatting manually:

```bash
ruff format .
ruff check --fix .
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-amazing-feature`)
3. Make your changes
4. Ensure all pre-commit hooks pass
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

