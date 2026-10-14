# how-llms-work

Interactive educational web app demonstrating key in LLM concepts and how LLMs work. Flask backend, vanilla JS frontend, Three.js for 3D visualization.

See [AGENTS.md](AGENTS.md) for architecture details, API contracts, testing patterns, and development lifecycle instructions.

## Quick Commands

```bash
make setup      # Create this worktree's Python 3.12 environment
make dev-start  # Start a persistent server on its allocated loopback/Tailscale port
make test       # Run pytest in this worktree's virtualenv
make dev-stop   # Stop only this worktree's server
```

## Key Files

- `app/routes/api.py` — all three REST endpoints
- `app/glove.py` — GloVe vector loader (required data file not in repo)
- `app/static/llm/` — frontend JS modules (no build step)
- `tests/test_api.py` — pytest suite; mock GloVe with `patch.dict`

## Setup

See [local development](docs/local-development.md). A `.env` file is optional.
Use `DEV_GLOVE_PATH` to select a host vector file; shared vectors are read-only input.
