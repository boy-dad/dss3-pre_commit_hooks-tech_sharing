# Tech-sharing
Documenting what works and what doesn't for efficient data science projects.


# Pre-commit Tools:

- [Pre-commit hooks](https://pre-commit.com/):
Pre-commit hooks are scripts or programs that run automatically before a developer's commit is processed. This allows developers to catch errors or style issues early, preventing them from being committed to the codebase. Pre-commit hooks can be used to enforce code style guidelines, run tests, or check for security vulnerabilities, among other things.
- [Black](https://black.readthedocs.io/en/stable/):
Black reformats code to follow a consistent style, removing the need for developers to manually format their code. Black is designed to be extremely configurable and has support for many different Python versions.
- [Flake8](https://flake8.pycqa.org/en/latest/)
Flake8 is a code quality checker for Python. It combines the functionality of three different tools: PyFlakes (which checks for logical errors), pycodestyle (which checks for code style issues), and McCabe (which checks for code complexity). Flake8 can be used to enforce coding standards and catch errors before they are committed.
- [iSort](https://pycqa.github.io/isort/)
isort is a Python utility that sorts imports alphabetically, separates them into sections, and automatically formats import statements. It can also remove unused imports and detect import formatting issues. isort is designed to be highly configurable and can be integrated with many different development tools.
- [nbstripout](https://github.com/kynan/nbstripout)
nbstripout is a command-line tool that removes output cells, metadata, and execution count information from Jupyter notebooks. It can be used to make notebooks more portable, as well as to remove sensitive information that may have been accidentally included in a notebook.
