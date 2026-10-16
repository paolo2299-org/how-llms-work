# How LLMs Work

An interactive Flask application explaining and demonstrating language-model
concepts.

## Local development

Development and CI use Python 3.12 with uv. Docker is used for production images
and deployment.

```bash
make worktree-create NAME=example-task
# cd to the printed path
make setup
make dev-start
make test
```

The server reports its allocated loopback or Tailscale URL and stays running
across chat endings. GloVe vectors are read from the worktree/main checkout's
`data/`, or from the existing shared VPS vector file. Use `DEV_GLOVE_PATH` to
select a different host file. The shared file is never modified or copied into
the worktree. Without vectors, page/tokenisation development works with
embeddings unavailable. Tests use mocked vectors.

When finished testing, run `make worktree-finish NAME=example-task` from a
retained checkout. It refuses unfinished/unpublished work and retains the
branch. See [local development](docs/local-development.md).

## Publishing images

Pushing to `main` runs `.github/workflows/deploy.yml`, which publishes a
versioned container image to GHCR. The VPS independently detects and deploys a
changed `main` digest using the root-owned configuration in the
`hetzner-vm-setup` repository; GitHub Actions has no SSH access to production.

Private native-development settings can be shared across worktrees without
committing secrets: see [private development configuration](docs/local-development.md#private-configuration-shared-by-worktrees).
Use `make dev-config` to validate the project's host config without printing values.
