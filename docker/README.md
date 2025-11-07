# Docker Guide for LLM_Criteria_Gemma

## Prerequisites

1. **Docker**: Install Docker Desktop or Docker Engine
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```

2. **NVIDIA Docker** (for GPU support):
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Docker Compose**: Usually included with Docker Desktop
   ```bash
   sudo apt-get install docker-compose-plugin
   ```

## Quick Start

### 1. Build Images

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build train
docker-compose build dev
```

### 2. Run Training

```bash
# Train with GPU
docker-compose up train

# Train with custom configuration
docker-compose run --rm train \
    python src/training/train_gemma_hydra.py \
    model.name=google/gemma-2-9b \
    training.batch_size=8
```

### 3. Development Environment

```bash
# Start Jupyter Lab
docker-compose up dev

# Access at http://localhost:8888
```

### 4. Run Tests

```bash
# Run all tests
docker-compose run --rm test

# Run specific tests
docker-compose run --rm test pytest tests/test_poolers.py -v
```

### 5. Evaluation

```bash
# Evaluate model
docker-compose run --rm eval \
    python src/training/evaluate.py \
    --checkpoint /workspace/outputs/fold_0/best_model.pt
```

### 6. Benchmarks

```bash
# Run pooler benchmarks
docker-compose up benchmark

# Custom benchmark
docker-compose run --rm benchmark \
    python benchmarks/benchmark_poolers.py \
    --device cuda \
    --batch-sizes 1 4 8 16 32
```

## Available Services

| Service | Description | GPU | Ports |
|---------|-------------|-----|-------|
| `train` | Training with 5-fold CV | ✅ | - |
| `dev` | Jupyter Lab development | ✅ | 8888, 6006 |
| `test` | Run test suite | ❌ | - |
| `eval` | Model evaluation | ✅ | - |
| `benchmark` | Performance benchmarks | ✅ | - |

## Docker Images

### Multi-stage Build Targets

1. **base**: CUDA + Python base
2. **dependencies**: + Python packages
3. **application**: + Application code
4. **training**: Production training
5. **development**: + Jupyter Lab
6. **cpu-only**: CPU-only lightweight image

### Image Sizes

| Image | Size (approx) | Purpose |
|-------|---------------|---------|
| `llm-criteria-gemma:train` | ~8GB | Training |
| `llm-criteria-gemma:dev` | ~9GB | Development |
| `llm-criteria-gemma:test` | ~2GB | Testing (CPU) |

## Volume Mounts

```yaml
volumes:
  - ./data:/workspace/data          # Dataset
  - ./outputs:/workspace/outputs    # Model checkpoints
  - ./logs:/workspace/logs          # Training logs
  - ./conf:/workspace/conf          # Hydra configs
```

## Environment Variables

```bash
# GPU configuration
NVIDIA_VISIBLE_DEVICES=all    # Use all GPUs
CUDA_VISIBLE_DEVICES=0,1      # Use specific GPUs

# Python configuration
PYTHONUNBUFFERED=1            # Unbuffered output
PYTHONDONTWRITEBYTECODE=1     # No .pyc files
```

## Common Tasks

### Interactive Shell

```bash
# GPU-enabled shell
docker-compose run --rm train /bin/bash

# Development shell
docker-compose run --rm dev /bin/bash
```

### Copy Files

```bash
# Copy from container
docker cp gemma-train:/workspace/outputs ./local_outputs

# Copy to container
docker cp ./local_data gemma-train:/workspace/data
```

### View Logs

```bash
# Follow logs
docker-compose logs -f train

# View specific service
docker-compose logs dev
```

### Resource Limits

```bash
# Limit GPU memory
docker-compose run --rm \
    -e CUDA_VISIBLE_DEVICES=0 \
    --gpus '"device=0"' \
    --shm-size=8gb \
    train
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Check runtime in daemon.json
cat /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

### Out of Memory

```bash
# Increase shared memory
docker-compose run --rm --shm-size=32gb train

# Clear cache
docker system prune -a
```

### Slow Build

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Build with no cache
docker-compose build --no-cache
```

## Production Deployment

### Build Production Image

```bash
docker build --target training -t llm-criteria-gemma:prod .
```

### Run Production Training

```bash
docker run -d \
    --name gemma-prod \
    --gpus all \
    --shm-size=16gb \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    llm-criteria-gemma:prod
```

### Monitor Training

```bash
# View logs
docker logs -f gemma-prod

# View GPU usage
watch -n 1 nvidia-smi

# Exec into container
docker exec -it gemma-prod /bin/bash
```

## Advanced Usage

### Multi-GPU Training

```bash
# Use all GPUs
docker-compose run --rm \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    train

# Distributed training (requires code changes)
docker-compose run --rm \
    -e MASTER_ADDR=localhost \
    -e MASTER_PORT=29500 \
    -e WORLD_SIZE=4 \
    -e RANK=0 \
    train
```

### Custom Configurations

Create `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  train:
    environment:
      - CUSTOM_VAR=value
    volumes:
      - /path/to/custom/data:/workspace/data
```

### CI/CD Integration

```bash
# Build in CI
docker build --target application -t $IMAGE_TAG .

# Run tests
docker run --rm $IMAGE_TAG pytest tests/

# Push to registry
docker push $IMAGE_TAG
```

## Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes
docker-compose down -v

# Clean up images
docker system prune -a

# Remove specific images
docker rmi llm-criteria-gemma:train
```

## Best Practices

1. **Use .dockerignore**: Exclude unnecessary files
2. **Multi-stage builds**: Smaller production images
3. **Layer caching**: Order Dockerfile commands by change frequency
4. **Security**: Don't run as root in production
5. **Logging**: Use Docker logging drivers
6. **Health checks**: Implement container health checks
7. **Resource limits**: Set memory and CPU limits

## Support

For issues:
- Check logs: `docker-compose logs`
- Verify GPU: `nvidia-smi`
- Review config: `docker-compose config`
- GitHub Issues: [Create issue](https://github.com/OscarTsao/LLM_Criteria_Gemma/issues)
