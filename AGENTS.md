# Agent Guide: how-llms-work

An interactive educational Flask app that demonstrates core LLM concepts. Where relevant these concepts are demonstrated through live visualizations in the browser.

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

## Development environments

For plain-language preview/start/stop/cleanup requests, use this repository's
`.agents/skills/dev-app/SKILL.md`. It is project-scoped on both the Mac and VPS.

Development and CI use a worktree-local Python 3.12 environment with uv. Docker
is reserved for production images and deployment. No production credentials are
needed for setup or the tests.

From a clean retained checkout, create one environment for a task:

```bash
make worktree-create NAME=example-task
# cd to the printed worktree path
make setup
make dev-start
make test
```

The helper creates branch `codex/example-task` in
`<main-checkout-name>-worktrees/example-task`. Reuse a task's suitable worktree
across turns. Each server has its own allocated port, log, control socket and
Playwright name recorded in `.dev/session.json`. Report the name, branch, path,
URL and Playwright session when handing it to the user.

The server runs independently of the launching command and remains available
after a turn ends or the phone disconnects. Use `make dev` only for an interactive
foreground terminal. Bind addresses are restricted to loopback or Tailscale;
`DEV_HOST=127.0.0.1 make dev-start` forces local-only access. Python changes need
a stop/start; static/template edits appear on refresh.

Use `make dev-status` for the URL and `make dev-stop` to stop only this worktree.
`make dev-prune` removes stale session metadata after a crash. After stopping,
`make dev-clean` also removes the development log. Use only the recorded named
Playwright session; close it when finished and never use `close-all`.

When the user says they are finished with a named environment, close its named
browser session and, from a retained checkout, run:

```bash
make worktree-check NAME=example-task
make worktree-finish NAME=example-task
```

Finish repeats the checks, stops only that server, removes the worktree and its
disposable logs/virtualenv/caches, and retains the branch. Ending an agent turn
does not authorize cleanup. Never force removal, discard work, delete branches,
or commit/push simply to bypass a guard without the user's authorization.
Report what was removed and the retained branch.

Only helper-created environments are eligible. Cleanup refuses tracked or
untracked changes, unpublished commits, locked/detached/switched worktrees and
unexpected ignored files such as `.env` or model checkpoints. It fetches
`origin` to verify that HEAD is preserved on a remote branch; fetch failure
also refuses cleanup. Do not run competing commands in the same environment.
Other environments remain independent. See `docs/local-development.md`.

Never run development commands against production Compose stacks, containers,
credentials or writable production data.

## Project development notes

The development adapter reads GloVe vectors without modifying them. It checks
`DEV_GLOVE_PATH` (or `GLOVE_PATH`), the worktree/main checkout's
`data/glove.6B.50d.txt`, then the existing VPS asset
`/srv/how-llms-work/data/glove.6B.50d.txt`. Treat the shared vector file as
read-only and never put development output in its directory. Native processes
run as the current user; this is read-only application usage, not a separate
filesystem sandbox.

Without a file, page/tokenisation development still works; embedding results
are unavailable and the launcher logs this limitation. An explicitly configured
missing path fails clearly. Tests use mocked vectors and tokenisation and need
no large asset downloads.

## Testing Patterns

Tests use `app.create_app(load_glove=False)` to skip the GloVe file dependency. The `glove_vectors` dict is patched inline via `patch.dict("app.glove.glove_vectors", fake, clear=True)`. Tiktoken is mocked with `MagicMock` to avoid network calls. Add new API tests following this pattern — do not test frontend JS.

## Deployment

GitHub Actions (`.github/workflows/deploy.yml`):
1. Run tests
2. Build Docker image
3. Push to GHCR (`ghcr.io/...`)

The workflow stops after publishing and has no SSH access to production. The VPS
independently pulls the public package and deploys changed `main` digests from a
root-owned Compose definition. Production uses gunicorn with 4 workers × 2
threads and mounts the GloVe data from `/srv/how-llms-work/data`.

## Key Constraints

- **No frontend build step** — JS files are loaded directly as ES modules via importmap in `index.html`. Do not introduce a bundler without updating the template.
- **Python 3.12** — specified in the Dockerfile base image.
