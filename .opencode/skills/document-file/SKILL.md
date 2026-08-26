---
name: document-file
description: Document per-method execution prerequisites (environment variables, external services, required files, initialization, infrastructure) in code docstrings so callers know what must be set up before a method runs.
compatibility: opencode
---

## Document Execution Prerequisites

For each function or method, determine whether anything must be configured, initialized, installed, running, or available before the method can execute successfully.

When prerequisites exist, document them in the docstring.

Look for requirements such as:

- Environment variables
- API keys or credentials
- Authentication
- External services
- Databases
- Model servers
- Network connectivity
- Required files or directories
- Configuration files
- Previously initialized objects
- Required application state
- Required repository or database records
- Hardware or devices
- Operating-system requirements
- Third-party executables or command-line tools

Infer prerequisites from the implementation and the methods it directly depends upon.

Do not invent prerequisites.

### Environment variables

If the method directly or indirectly depends on environment variables, document them by name and explain their purpose.

Example:

```python
def generate_response(self, prompt: str) -> str:
    """Generate a response using the configured LLM provider.

    Setup:
        - `OPENAI_API_KEY` must contain a valid OpenAI API key.
        - Network access to the configured OpenAI endpoint is required.

    Args:
        prompt: Prompt to send to the model.

    Returns:
        The generated response.
    """
````

Never include the actual value of secrets.

### External services

Document services that must already be available.

Example:

```python
def search_documents(self, query: str) -> list[Document]:
    """Search indexed documents.

    Setup:
        - Elasticsearch must be running and reachable.
        - `ELASTICSEARCH_URL` must identify the Elasticsearch endpoint.
        - The document index must already exist.

    Args:
        query: Search expression.

    Returns:
        Documents matching the query.
    """
```

### Required initialization

Document application state that must exist before calling the method.

Example:

```python
def execute(self, command: Command) -> Result:
    """Execute a command using the registered components.

    Setup:
        - The component registry must be initialized.
        - Required components must be registered before this method is called.

    Args:
        command: Command to execute.

    Returns:
        Result of the command execution.
    """
```

### Required files

Document files or directories that must exist.

Example:

```python
def load_policy(self) -> BrandingPolicy:
    """Load the branding policy.

    Setup:
        - `config/branding.yaml` must exist and be readable.

    Returns:
        The configured branding policy.
    """
```

### Local infrastructure

Document locally running infrastructure when required.

Example:

```python
def run_agent(self, prompt: str) -> AgentResult:
    """Run the agent using the configured local model.

    Setup:
        - Ollama must be running.
        - The configured model must already be available to Ollama.
        - `OLLAMA_BASE_URL` may be used to configure the Ollama endpoint.

    Args:
        prompt: User request to process.

    Returns:
        Result produced by the agent.
    """
```

## Setup Documentation Rules

* Add a `Setup:` section only when prerequisites actually exist.
* Document requirements necessary to execute the method, not general project setup.
* Prefer exact environment-variable, service, file, and configuration names found in the code.
* Trace direct dependencies when necessary to discover hidden prerequisites.
* Do not document speculative requirements.
* Never expose passwords, tokens, API keys, or other secret values.
* Explain required initialization or state when callers must perform another operation first.
* Keep setup instructions local to the method being documented.
