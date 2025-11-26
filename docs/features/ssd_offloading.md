# SSD Offloading with GPU Direct Storage (Experimental)

This page introduces the SSD KV cache offloading feature in vLLM using NVIDIA's GPU Direct Storage (GDS).

!!! note
    This feature is experimental and subject to change.

## Overview

SSD offloading allows vLLM to use NVMe SSDs as an extended cache for KV (key-value) data. When GPU memory is insufficient to hold all the KV cache, vLLM can offload less recently used blocks to SSD storage and reload them when needed.

The key advantage of using GPU Direct Storage (GDS) is that data transfers happen directly between GPU memory and the SSD, bypassing CPU memory entirely. This results in:

- **Lower latency**: No CPU memory copy overhead
- **Higher throughput**: Direct DMA transfers between GPU and NVMe
- **Reduced CPU utilization**: CPU is not involved in data movement

## Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, or newer)
- **Storage**: NVMe SSD (PCIe Gen3 x4 or better recommended)
- **System**: PCIe topology that allows direct GPU-to-NVMe communication

### Software Requirements

- **CUDA**: 11.4 or newer with GDS support
- **cuFile Library**: Part of CUDA Toolkit or available via nvidia-gds package
- **Filesystem**: ext4 or XFS (GDS-compatible filesystems)
- **kvikio**: RAPIDS I/O library (pip install kvikio-cu12)

!!! tip "Installing kvikio"
    ```bash
    # For CUDA 12.x
    pip install kvikio-cu12

    # For CUDA 11.x
    pip install kvikio-cu11
    ```

### GDS Setup

1. **Install GDS drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nvidia-gds

   # Or install from CUDA Toolkit
   sudo apt-get install cuda-gds-12-x
   ```

2. **Mount filesystem with DAX** (optional, for best performance):
   ```bash
   sudo mount -o dax /dev/nvme0n1p1 /mnt/nvme
   ```

3. **Verify GDS is working**:
   ```python
   import kvikio
   props = kvikio.DriverProperties()
   print(f"GDS available: {props.is_gds_available}")
   ```

## Usage

### Basic Usage

Enable SSD offloading with the `--kv-transfer-config` option:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SSDOffloadingSpec",
            "num_ssd_blocks": 10000,
            "ssd_cache_dir": "/mnt/nvme/vllm_cache"
        }
    }'
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `spec_name` | string | - | Must be `"SSDOffloadingSpec"` for SSD offloading |
| `num_ssd_blocks` | int | Required | Number of KV cache blocks to allocate on SSD |
| `ssd_cache_dir` | string | `/tmp/vllm_ssd_cache` | Directory path for SSD cache files |
| `cache_file_prefix` | string | `"kv_cache"` | Prefix for cache file names |
| `eviction_policy` | string | `"lru"` | Cache eviction policy: `"lru"` or `"arc"` |
| `num_io_threads` | int | `4` | Number of I/O threads for async operations |
| `block_size` | int | GPU block size | Offloaded block size in tokens |

### Advanced Configuration

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SSDOffloadingSpec",
            "num_ssd_blocks": 50000,
            "ssd_cache_dir": "/mnt/nvme/vllm_cache",
            "block_size": 64,
            "eviction_policy": "arc",
            "num_io_threads": 8
        }
    }'
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Scheduler                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Offloading Manager                      │   │
│  │  - Tracks which blocks are on GPU vs SSD            │   │
│  │  - Manages LRU/ARC eviction policy                  │   │
│  │  - Coordinates load/store operations                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      vLLM Worker                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           GpuSsdOffloadingHandler                    │   │
│  │  - Uses kvikio/cuFile for GDS transfers             │   │
│  │  - Async I/O with thread pool                       │   │
│  │  - Manages cache files on SSD                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                   ┌───────────────┐
    │   GPU Memory  │  ◄── GDS ──►      │   NVMe SSD    │
    │   (KV Cache)  │                   │ (Cache Files) │
    └───────────────┘                   └───────────────┘
```

### Data Flow

1. **Offloading (GPU → SSD)**:
   - When GPU KV cache is full, the scheduler identifies blocks to evict
   - Worker uses GDS to write blocks directly from GPU to SSD
   - No CPU memory is used for the transfer

2. **Loading (SSD → GPU)**:
   - When a request needs previously offloaded KV data
   - Worker uses GDS to read blocks directly from SSD to GPU
   - Request continues with restored KV cache

### Eviction Policies

- **LRU (Least Recently Used)**: Evicts the least recently accessed blocks first. Good for workloads with clear temporal locality.

- **ARC (Adaptive Replacement Cache)**: Automatically adapts between recency and frequency. Better for mixed workloads.

## Performance Considerations

### SSD Selection

- Use NVMe SSDs with high sequential read/write performance
- PCIe Gen4 or Gen5 SSDs provide better throughput
- Enterprise SSDs with power-loss protection are recommended for production

### Cache Directory Location

- Place cache directory on a fast NVMe SSD (not HDD or network storage)
- Use a dedicated partition or SSD for cache to avoid interference
- Ensure sufficient free space: `num_ssd_blocks * block_size * bytes_per_block`

### Block Size Tuning

- Larger block sizes reduce I/O overhead but may waste space
- Smaller block sizes are more granular but have higher overhead
- Default GPU block size is usually optimal for most workloads

### Monitoring

Check GDS statistics:
```python
import kvikio
props = kvikio.DriverProperties()
print(f"GDS available: {props.is_gds_available}")
print(f"NVIDIA driver version: {props.nvfs_driver_version}")
```

## Comparison with CPU Offloading

| Aspect | CPU Offloading | SSD Offloading (GDS) |
|--------|---------------|---------------------|
| Latency | Lower | Higher |
| Capacity | Limited by RAM | Limited by SSD size |
| Throughput | ~20-50 GB/s | ~5-15 GB/s |
| CPU Usage | Higher | Minimal |
| Cost | RAM is expensive | NVMe is cheaper |

**When to use SSD offloading:**
- When you need more capacity than available CPU RAM
- When CPU RAM is needed for other tasks
- For long-context or batch workloads where latency is less critical

**When to use CPU offloading:**
- For latency-sensitive workloads
- When sufficient CPU RAM is available
- For smaller models with frequent KV cache access

## Troubleshooting

### GDS Not Available

If you see "GDS driver not available" warnings:

1. Check if nvidia-gds package is installed
2. Verify cuFile library is accessible
3. Check filesystem compatibility (ext4/XFS)
4. Try running with `KVIKIO_COMPAT_MODE=on` for compatibility mode

### Poor Performance

1. Verify SSD is NVMe (not SATA)
2. Check PCIe bandwidth between GPU and SSD
3. Ensure cache directory is on NVMe, not network storage
4. Increase `num_io_threads` for higher parallelism

### Cache Files Not Cleaned Up

Cache files are automatically cleaned up when the server shuts down gracefully. If files remain:

```bash
rm -rf /mnt/nvme/vllm_cache/kv_cache_*.bin
```

## Example: Long Context Serving

For serving models with very long contexts (100K+ tokens), SSD offloading can significantly increase effective batch size:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 128000 \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SSDOffloadingSpec",
            "num_ssd_blocks": 100000,
            "ssd_cache_dir": "/mnt/nvme/vllm_cache",
            "eviction_policy": "arc"
        }
    }'
```

This configuration allows serving many long-context requests by offloading older KV cache blocks to SSD, freeing GPU memory for new requests.
