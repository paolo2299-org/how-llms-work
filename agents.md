# Agent Guide: how-llms-work

An interactive educational Flask app that demonstrates three core LLM concepts — tokenization, word embeddings, and self-attention — through live visualizations in the browser.

## Architecture

```
app/
├── __init__.py          # App factory: create_app(load_glove=True)
├── config.py            # Dev/prod config; GLOVE_PATH env var
├── glove.py             # Loads GloVe vectors into module-level dict at startup
├── routes/
│   ├── pages.py         # GET / → index.html
│   └── api.py           # REST endpoints (see below)
├── static/
│   └── llm/             # Vanilla JS ES modules (no build step)
│       ├── llm.js           # Entry point; imports tokeniser + attention
│       ├── tokeniser.js     # Calls /api/tokenize, renders token chips
│       ├── embeddings.js    # Three.js 3D scene; calls /api/embed
│       ├── embedding-model.js  # API layer + word validation for embeddings
│       ├── attention.js     # Hardcoded attention weights; interactive heatmap
│       └── messages.js      # UI toast/message utilities
└── templates/
    └── index.html       # Single-page app; importmap for Three.js
tests/
└── test_api.py          # 11 pytest tests for all API endpoints
```

## API Endpoints

| Method | Path | Input | Output |
|--------|------|-------|--------|
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/api/tokenize?text=...` | query param `text` | `{"tokens": [...], "token_ids": [...]}` |
| POST | `/api/embed` | JSON `{"words": [...]}` | `{"points": [{word, x, y, z}, ...], "unknown": [...]}` |

**Tokenization** uses `tiktoken` with the `cl100k_base` encoding (same as GPT-4).

**Embeddings** uses GloVe 50-dimensional vectors, then PCA-reduces to 3D:
- 1 word → origin `(0, 0, 0)`
- 2 words → 1-component PCA, y/z padded to 0
- 3+ words → up to 3-component PCA

## GloVe Data File

The app requires `glove.6B.50d.txt` at the path set by `GLOVE_PATH` (default: `/app/data/glove.6B.50d.txt`). This file is not in the repo (`.gitignore` excludes `data/*.txt`). Without it the embed endpoint returns empty results. In tests, `create_app(load_glove=False)` skips loading it.

## Development Workflow

```bash
make dev          # Docker Compose up with Flask debug mode, port 8080
make test         # Run pytest suite in Docker
make shell        # Interactive shell in dev container
make down         # Tear down containers
```

Copy `.env.example` to `.env` and set `GLOVE_PATH` before running locally.

## Testing Patterns

Tests use `app.create_app(load_glove=False)` to skip the GloVe file dependency. The `glove_vectors` dict is patched inline via `patch.dict("app.glove.glove_vectors", fake, clear=True)`. Tiktoken is mocked with `MagicMock` to avoid network calls. Add new API tests following this pattern — do not test frontend JS.

## Deployment

GitHub Actions (`.github/workflows/deploy.yml`):
1. Run tests
2. Build Docker image
3. Push to GHCR (`ghcr.io/...`)
4. SSH deploy to production VM using `compose.prod.yml`

Production uses gunicorn with 4 workers × 2 threads. Secrets live in GitHub Actions secrets/vars.

## Key Constraints

- **No frontend build step** — JS files are loaded directly as ES modules via importmap in `index.html`. Do not introduce a bundler without updating the template.
- **GloVe lookup is case-insensitive** — `api.py` lowercases every word before lookup; the original casing is preserved in the response.
- **Single-page app** — all three demos live in `index.html`; there are no separate routes for them.
- **Python 3.12** — specified in the Dockerfile base image.
