# How LLMs Work

An interactive Flask application explaining and demonstrating language-model
concepts.

## Local development

Copy the example environment, make sure the GloVe data exists under `data/`,
and start the development container:

```bash
cp .env.example .env
make dev
```

The app is available at `http://localhost:18083`. Development always uses the
`how-llms-work-dev` Compose project and publishes only on the host loopback
interface, keeping its container, network, and configuration separate from
production. Run tests with `make test` and stop the development containers with
`make down`.

On the VPS, the existing GloVe data can be reused through a read-only mount by
setting this in `.env`:

```bash
DEV_DATA_DIR=/srv/how-llms-work/data
```

From a Mac, forward the development port and open `http://localhost:18083`:

```bash
ssh -N -L 18083:127.0.0.1:18083 deploy@hetzner
```

Set `DEV_PORT` in `.env` to override the default port when necessary.

## Publishing images

Pushing to `main` runs `.github/workflows/deploy.yml`, which publishes a
versioned container image to GHCR. The VPS independently detects and deploys a
changed `main` digest using the root-owned configuration in the
`hetzner-vm-setup` repository; GitHub Actions has no SSH access to production.
