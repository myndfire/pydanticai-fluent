---
name: document-folder
description: Analyze the files in a folder and create or update a README.md describing what the code does, its requirements, configuration, expected output, and how to run it.
compatibility: opencode
---

# Folder Documentation

Analyze the requested folder and create or update `README.md` in that folder.

## Goal

The README must allow a developer unfamiliar with the folder to understand:

- What the code does
- Which file or files are intended to be executed
- What runtime and dependencies are required
- What external services are required
- What environment variables can be configured
- How to configure the environment
- How to run the code
- What output or behavior to expect

## Workflow

### 1. Inspect the folder

Examine all relevant files in the requested folder.

Include:

- Source files
- `pyproject.toml`
- `requirements.txt`
- `package.json`
- lock files
- configuration files
- `.env.example`
- `.env.sample`
- shell scripts
- container/configuration files
- existing README files
- supporting modules imported by executable files

Ignore generated or irrelevant directories such as:

- `.git`
- `.venv`
- `venv`
- `node_modules`
- `__pycache__`
- `.pytest_cache`
- `dist`
- `build`

Do not recursively analyze unrelated projects outside the requested folder.

### 2. Identify executable files

Determine which files appear to be entry points.

Examples:

- Python files containing `if __name__ == "__main__"`
- CLI applications
- scripts
- server startup modules
- files referenced by package scripts
- shell scripts intended to be executed

If multiple runnable examples exist, document each one separately.

Do not assume that every source file is independently executable.

Record the **exact basename** of each entry-point file as it appears in the folder
(e.g., `01_quality_check.py`). Use these real names in the README — never a
generic placeholder such as `main.py` or `app.py`.

### 3. Understand what the code does

Read the implementation sufficiently to explain:

- Purpose of the program
- Major components
- Execution flow
- Important libraries
- External systems or APIs
- Inputs
- Outputs
- Side effects

Describe behavior from the source code rather than merely repeating file names or comments.

### 3b. Capture per-file details

For each entry-point or key source file identified in Steps 2–3, record enough
detail to write a thorough description. For every file, determine:

- **Purpose:** what the file demonstrates or is responsible for
- **Key components:** classes, functions, or modules it defines or exercises
- **Key configuration and APIs:** builder/config classes and the methods it uses
- **Notable behaviors:** callbacks, guards, handlers, or side effects
  (e.g., `on_retry`, `on_error`, `on_filter` handlers)
- **Representative inputs:** demo prompts, arguments, or example data it runs

Describe from the source code, not merely from comments or filenames.

### 4. Determine setup requirements

Identify runtime and dependency requirements from the repository.

Examples:

- Python version
- Node.js version
- package manager
- `uv`
- pip
- npm
- pnpm
- required services
- databases
- local model servers
- API providers

Prefer information explicitly present in configuration files.

Do not invent versions or dependencies that cannot be established from the project.

### 5. Analyze environment variables

Search the source code and configuration for environment-variable usage.

Look for patterns such as:

- `os.getenv`
- `os.environ`
- `BaseSettings`
- `pydantic-settings`
- `dotenv`
- `process.env`
- configuration wrappers
- `.env.example`
- `.env.sample`

For every discovered environment variable determine, where possible:

- Variable name
- Purpose
- Whether it is required
- Default value
- Example value
- Which component uses it

Never expose secret values from an actual `.env` file.

If an actual `.env` exists, use it only to determine variable names when necessary. Never copy passwords, tokens, API keys, credentials, or other secret values into the README.

Prefer `.env.example` and source-code definitions.

### 6. Determine how to run the code

Infer the canonical run command from project configuration.

Use the **exact filename** discovered in Step 2. For example, if the entry point
is `01_quality_check.py`, the command is:

```bash
uv run python 01_quality_check.py
```

Other valid forms, using the real filename:

```bash
python 01_quality_check.py
```

```bash
npm run dev
```

```bash
uvicorn app.main:app
```

Prefer commands supported directly by the repository configuration.

Reference files by their exact name as it appears in the target folder — never a
generic placeholder like `main.py` or `app.main:app`.

Do not invent a run command when the correct command cannot be established.

### 7. Determine expected output

Inspect:

* print statements
* logging calls
* HTTP endpoints
* return structures
* generated files
* agent responses
* examples
* tests

Describe what the developer should expect after running the program.

Use representative examples when they can be established from the source.

Do not claim exact output when output is dynamic.

### 8. Create README.md

Create or update:

`<target-folder>/README.md`

Use this structure where applicable:

# <Folder or Application Name>

Brief description of the application.

## Overview

Detailed explanation of what the code does and its execution flow.

## Files

For each important file, add a subsection using the file's **exact basename** as
the heading (e.g., `### 01_quality_check.py`). In each subsection, describe in
prose:

- What the file demonstrates or is responsible for
- Key components, configuration classes, or methods it uses
- Notable callbacks, behaviors, or guards it exercises
- The representative inputs or demo prompts it runs (when applicable)

Example:

### 01_quality_check.py

Demonstrates quality validation of agent output. Builds `QualityConfig()
.with_model(...).with_retries(...)` and registers an `on_error` callback that
returns a graceful fallback. Demo prompt: "Check the response for
hallucinations." Prints `success`, `output`, and `error_context`.

List each executable or key source file by its exact name as it appears in the
folder, not a generic name.

## Prerequisites

List required runtimes, tools, services, and dependencies.

## Setup

Provide installation and preparation instructions.

## Configuration

Document environment variables in a table:

| Variable | Required | Default | Description |
| -------- | -------- | ------- | ----------- |

Never include real credentials or secrets.

Include a `.env` example containing placeholders when useful.

## Running

Provide exact commands for running the application. Use the **exact filename**
of each entry point as it appears in the folder (e.g., `uv run 01_quality_check.py`).

If multiple entry points exist, document each one.

## Expected Output

Explain what should happen when the application runs.

Include representative console output or behavior when it can be determined from the source.

## How It Works

Explain the main execution flow step by step.

Focus on architecture and important interactions rather than explaining every line of code.

## Troubleshooting

Include obvious setup problems that can be inferred from the implementation, such as:

* missing environment variables
* unreachable external services
* missing dependencies
* missing model servers
* incorrect ports

Only document problems supported by the source or configuration.

## Rules

* Inspect source code before writing documentation.
* Do not infer behavior solely from filenames.
* Do not invent dependencies.
* Do not invent environment variables.
* Do not expose secrets.
* Do not execute applications merely to determine their behavior.
* Do not modify source files.
* Only create or modify `README.md`.
* Preserve useful manually-written README content when updating an existing README.
* Prefer commands and configuration already used by the project.
* Clearly state when something cannot be determined from the source.

```
