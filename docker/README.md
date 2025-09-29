# Docker images

Two build targets are provided so you can run the pipelines without cloning the repo each time.

## Images

| Tag | Dockerfile | Entry point | Default command |
| --- | --- | --- | --- |
| `whale` | `Dockerfile.whale` | `whale-deep-dive` | `--help` |
| `whale-fast` | `Dockerfile.fast` | `whale-deep-dive-fast` | `--help` |

## Build locally

```powershell
# From the repository root
docker build -f docker/Dockerfile.whale -t whale .
docker build -f docker/Dockerfile.fast -t whale-fast .
```

Because the tags are short (`whale`, `whale-fast`), you can run them without a namespace:

```powershell
docker run --rm whale --synthetic --methods random --max-points 2000
docker run --rm whale-fast --synthetic --methods hybrid --m 900 --max-points 60000
```

## Publishing to a registry

- **GitHub Container Registry** (recommended):
  ```powershell
  docker tag whale ghcr.io/jorgeLRW/whale:0.1.0
  docker tag whale-fast ghcr.io/jorgeLRW/whale-fast:0.1.0
  docker push ghcr.io/jorgeLRW/whale:0.1.0
  docker push ghcr.io/jorgeLRW/whale-fast:0.1.0
  ```
- **Docker Hub**: replace the registry prefix with your Docker Hub username (`docker tag whale yourname/whale:0.1.0`).

## Keeping images slim

- Use `docker build --pull` to grab the latest patched base image.
- Rebuild whenever crucial dependencies (NumPy, SciPy, FAISS) receive security updates.
- Consider `docker scout cves whale` to scan for vulnerabilities periodically.
