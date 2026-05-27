# how-llms-work

Interactive educational web app demonstrating tokenization, word embeddings, and self-attention in LLMs. Flask backend, vanilla JS frontend, Three.js for 3D visualization.

See [agents.md](agents.md) for architecture details, API contracts, testing patterns, and deployment notes.

## Quick Commands

```bash
make dev     # Start dev server on port 8080 (Docker, hot reload)
make test    # Run test suite (pytest in Docker)
make down    # Stop containers
```

## Key Files

- `app/routes/api.py` — all three REST endpoints
- `app/glove.py` — GloVe vector loader (required data file not in repo)
- `app/static/llm/` — frontend JS modules (no build step)
- `tests/test_api.py` — pytest suite; mock GloVe with `patch.dict`

## Setup

Copy `.env.example` to `.env`. Set `GLOVE_PATH` to your local `glove.6B.50d.txt`.
