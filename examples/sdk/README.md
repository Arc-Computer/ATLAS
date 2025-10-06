# Atlas SDK Runtime Examples

These examples pair with the SDK Runtime documentation to show how to run, customize, and debug `atlas.run`.

## Prerequisites

1. Clone the SDK and install in editable mode:
   ```bash
   git clone https://github.com/Arc-Computer/atlas-sdk
   cd atlas-sdk
   pip install -e .
   ```
2. Export an API key compatible with the configuration you plan to use (the quickstart examples rely on `OPENAI_API_KEY`).

## Example Index

| File | Description |
|------|-------------|
| `01_minimal_example.py` | Runs the quickstart config end-to-end and prints the final answer. |
| `02_custom_adapter.py` | Spins up a tiny HTTP echo service and wraps it with the HTTP adapter. |
| `03_error_handling.py` | Shows how to catch adapter errors and inspect retry metadata. |

## Running an Example

From the root of the SDK repo:

```bash
python examples/sdk/01_minimal_example.py
```

Each script includes inline comments guiding you to the relevant docs page for deeper context:

- [`sdk/quickstart`](../../docs/sdk/quickstart.mdx)
- [`sdk/adapters`](../../docs/sdk/adapters.mdx)
- [`sdk/orchestration`](../../docs/sdk/orchestration.mdx)
