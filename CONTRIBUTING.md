# Contributing to `twilio-agent-connect-python`

We'd love for you to contribute to our source code and to make `twilio-agent-connect-python` even better than it is today! Here are the guidelines we'd like you to follow:

- [Code of Conduct](#code-of-conduct)
- [Question or Problem?](#got-an-apiproduct-question-or-problem)
- [Issues and Bugs](#found-an-issue)
- [Feature Requests](#want-a-feature)
- [Documentation Fixes](#want-a-doc-fix)
- [Submission Guidelines](#submission-guidelines)
- [Coding Rules](#coding-rules)

## Code of Conduct

Help us keep this project open and inclusive. Please be kind and considerate of other developers, and treat all community members with respect.

## Got an API/Product Question or Problem?

If you have questions about how to use the SDK, please check out the [README](README.md) and [Getting Started guide](getting_started/README.md) first.

If you still need help, reach out to [Twilio Support](https://www.twilio.com/help/contact). GitHub issues are reserved for bug reports and feature requests, not general support questions.

## Found an Issue?

If you find a bug in the source code or a mistake in the documentation, you can help us by submitting an issue to our [GitHub Repository][github]. Even better, you can submit a Pull Request with a fix.

## Want a Feature?

You can request a new feature by submitting an issue to our [GitHub Repository][github].

If you would like to implement a new feature then consider what kind of change it is:

- **Major Changes** should be discussed first in an issue so that we can coordinate efforts, prevent duplication of work, and help you craft the change so that it is successfully accepted into the project.
- **Small Changes** can be crafted and submitted to the [GitHub Repository][github] as a Pull Request.

## Want a Doc Fix?

If you want to help improve the docs, create an issue or submit a Pull Request with your proposed changes.

## Submission Guidelines

### Submitting an Issue

Before you submit your issue, search the archive — maybe your question was already answered.

If your issue appears to be a bug, and hasn't been reported, open a new issue. Help us maximize the effort we can spend fixing issues and adding new features by not reporting duplicate issues. Providing the following information will increase the chances of your issue being dealt with quickly:

- **Overview of the Issue** - if an error is being thrown, include the stack trace
- **Motivation / Use Case** - explain why this is a bug for you
- **SDK Version** - which version of the SDK are you using?
- **Python Version** - which version of Python are you running? (must be 3.10+)
- **Operating System** - if relevant
- **Reproduce the Error** - provide steps or an isolated code snippet that reproduces the issue
- **Related Issues** - has a similar issue been reported before?
- **Suggest a Fix** - if you can't fix the bug yourself, perhaps you can point to what might be causing the problem

### Submitting a Pull Request

Before you submit your Pull Request (PR) consider the following:

1. Search [GitHub](https://github.com/twilio/twilio-agent-connect-python/pulls) for an open or closed PR that relates to your submission. You don't want to duplicate effort.

2. Fork the repo and create a new branch from `main`:

   ```shell
   git checkout -b my-fix-branch main
   ```

3. Set up your development environment:

   ```shell
   make dev-setup
   ```

4. Make your changes, **including appropriate test cases**.

5. Follow our [Coding Rules](#coding-rules).

6. Run the full validation suite and ensure all checks pass:

   ```shell
   make check
   ```

7. Commit your changes with a descriptive commit message.

8. Push your branch to GitHub:

   ```shell
   git push origin my-fix-branch
   ```

9. Open a Pull Request against `main` and fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

After your pull request is merged, you can safely delete your branch and pull the changes from the main (upstream) repository.

## Coding Rules

To ensure consistency throughout the source code, keep these rules in mind as you are working:

- **Tests** - All features or bug fixes must be tested. Write tests using [pytest](https://docs.pytest.org/).
- **Type Safety** - Code must pass `make type-check` (mypy with the project's configured strictness settings).
- **Linting** - Code must pass `make lint` (ruff).
- **Formatting** - Code must pass `make format` (ruff format — line length 100). Run `make format` to auto-fix.
- **Python Version** - Target Python 3.10+. Use modern syntax (type unions with `|`, etc.).
- **Validation** - Use [Pydantic](https://docs.pydantic.dev/) for runtime validation and type inference.

[github]: https://github.com/twilio/twilio-agent-connect-python
