# keycycle

Thread-safe key rotation and rate limiting manager for LLM API keys.
Supports OpenAI-compatible providers and Agno models.
Persists usage data to SQL (MySQL/TiDB).

## Installation

```bash
pip install keycycle
# Optional extras
pip install keycycle[openai]
pip install keycycle[agno]
pip install keycycle[all]
```

## Configuration

Define API keys in a `.env` file.
Format: `provider_API_KEY_index`.
Also specify the count with `NUM_provider`.

```ini
# .env
NUM_OPENAI=2
OPENAI_API_KEY_1=sk-...
OPENAI_API_KEY_2=sk-...

# Optional: DB Connection (Defaults to TIDB_DB_URL env var)
TIDB_DB_URL=mysql+pymysql://user:pass@host:port/db
```

Supported providers for auto-loading: `OPENROUTER`, `GEMINI`, `CEREBRAS`, `GROQ`.

## Usage

### OpenAI Client (Sync & Async)

Wraps the standard `openai` library. Drop-in replacement.

```python
import os
from keycycle import MultiProviderWrapper

# 1. Initialize Wrapper
wrapper = MultiProviderWrapper.from_env(
    provider="openai",
    default_model_id="gpt-4o",
    db_url=os.getenv("DATABASE_URL") 
)

# 2. Get Rotating Client (Standard OpenAI Interface)
client = wrapper.get_openai_client(estimated_tokens=500)

# 3. Use standard methods
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o"
)
print(response.choices[0].message.content)

# Async
async_client = wrapper.get_async_openai_client()
# await async_client.chat.completions.create(...)
```

### Agno Integration

Wraps Agno models with rotation logic.

```python
from keycycle import MultiProviderWrapper

wrapper = MultiProviderWrapper.from_env(provider="openai", default_model_id="gpt-4o")

# Returns a model instance with rotation mixin
model = wrapper.get_model(
    id="gpt-4o",
    instructions="You are a bot."
)

model.generate("Hello world")
```

### Statistics

Print usage stats to console (uses `rich`).

```python
wrapper.print_global_stats()
wrapper.print_key_stats(0) # Stats for key index 0
wrapper.print_model_stats("gpt-4o")
```

## Features

*   **Rotation:** Round-robin selection. Skips keys on cooldown.
*   **Rate Limiting:** Enforces RPM, TPM, RPD, TPD limits.
*   **Failover:** Auto-rotates on `429 Too Many Requests`.
*   **Persistence:** Logs usage to SQL database for historical tracking.
*   **Thread-Safe:** Safe for concurrent usage.

## Development and Publishing

The project includes scripts to automate the build and release process to PyPI.

### Publishing Scripts

*   `publish.sh`: Bash script for Linux/macOS.
*   `publish.ps1`: PowerShell script for Windows.

These scripts perform the following:
1. Load `PYPI_TOKEN` from the `.env` file.
2. Clean `dist/`, `build/`, and `*.egg-info`.
3. Build the package using `python -m build`.
4. Upload to PyPI using `twine`.

### PyPI Configuration

To publish, ensure your `.env` file contains your PyPI API token:

```ini
# .env
PYPI_TOKEN=pypi-AgEIcHlwaS5vcmc...
```

The scripts automatically map this to `TWINE_PASSWORD` and set `TWINE_USERNAME` to `__token__`.

## Database Schema

The library uses SQLAlchemy to manage a `usage_logs` table.
Ensure your database user has `CREATE` and `INSERT` permissions.
Designed for TiDB but works with standard MySQL.
