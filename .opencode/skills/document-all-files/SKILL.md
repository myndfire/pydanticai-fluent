---
name: document-all-files
description: Apply the document-file prerequisite-documentation methodology to every Python source file in a folder, adding `Setup:` sections to function/method docstrings that have execution prerequisites.
compatibility: opencode
---

# Document All Files

Apply the `document-file` prerequisite-documentation approach to every Python
source file in a target folder. For each file, analyze its functions and methods
and add `Setup:` docstring sections wherever execution prerequisites exist.

## Input

You will be given a target folder (an absolute or workspace-relative path). If no
folder is provided, ask the user for one before proceeding.

## Workflow

### 1. Enumerate Python files (non-recursive)

List the `*.py` files directly inside the target folder. Do **not** recurse into
subfolders.

Ignore generated or irrelevant paths (including any that happen to be nested):
- `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.pytest_cache`,
  `dist`, `build`

Process each `*.py` file independently. Do not process `README*` files,
configuration files, or lock files.

### 2. Apply the document-file methodology per file

For every Python file, follow the same rules as the `document-file` skill
(canonical spec: `.opencode/skills/document-file/SKILL.md`). In summary:

- For each function/method in the file, determine whether anything must be
  configured, initialized, installed, running, or available before it can
  execute successfully.
- When prerequisites exist, add a `Setup:` section to that function's docstring
  documenting them.
- Look for: environment variables / API keys, external services, databases,
  model servers, network connectivity, required files/directories,
  configuration files, previously initialized objects/state, OS or hardware
  requirements, and third-party executables.
- Infer prerequisites from the implementation and its direct dependencies. Do not
  invent them.

#### Setup section format

```python
def example(self, x):
    """Short description.

    Setup:
        - `<ENV_VAR>` must contain a valid value and explain its purpose.
        - `<Service>` must be running and reachable.
        - `<file_or_dir>` must exist and be readable.

    Args:
        x: ...

    Returns:
        ...
    """
```

#### Rules

- Add a `Setup:` section only when prerequisites actually exist.
- Document requirements necessary to execute the method, not general project
  setup.
- Prefer exact environment-variable, service, file, and configuration names
  found in the code.
- Trace direct dependencies to discover hidden prerequisites.
- Do not document speculative requirements.
- Never expose passwords, tokens, API keys, or other secret values.
- Keep setup instructions local to the method being documented.

### 3. Skip already-documented files

If a file's functions already contain adequate `Setup:` documentation for their
real prerequisites, do not modify it. Only update a file when a genuine
prerequisite is missing or incorrect.

### 4. Report

When finished, give a brief summary: which files were updated, which were skipped
(already documented or no functions with prerequisites), and any files where
prerequisites could not be determined from the source.

## Constraints

- Only edit Python source files' docstrings (add or extend `Setup:` sections).
- Do not rewrite unrelated logic.
- Do not invent dependencies or environment variables.
- Do not expose secrets.
- Only create or modify files within the target folder.
