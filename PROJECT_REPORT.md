# SSD-Backed KV Cache PagedAttention for vLLM

## Project Report

**Course**: [Your Course Name]  
**Team Members**: [Your Names]  
**Date**: December 2025

---

## 1. Problem Statement and Motivation

### 1.1 The Problem We Addressed

PagedAttention, a key component of modern Large Language Model (LLM) serving systems like vLLM, stores key-value (KV) tensors in non-contiguous, fixed-size blocks ("pages") to achieve high throughput by reducing memory fragmentation and enabling fine-grained reuse across requests. However, **the current system does not implement a fast "swap" path to secondary storage**—when GPU memory fills up, blocks are reclaimed and future reuse opportunities are permanently lost.

Currently, vLLM only offloads excess KV cache to CPU memory. We propose to add an **SSD storage tier for KV blocks** and enable seamless data movement using **GPUDirect Storage**. Specifically, we aim to implement SSD-backed PagedAttention where blocks of KV cache can be evicted to and fetched from NVMe SSDs **directly to/from GPU memory, bypassing system memory and CPU overhead**. This maximizes KV reuse without stalls or discarding useful cache.

### 1.2 Why We Selected This Problem

#### Economic Motivation: SSDs Provide the Biggest Bang for the Buck

The current solutions for expanding KV cache capacity are expensive:

| Solution | Cost | Drawbacks |
|----------|------|-----------|
| More GPU memory | $$$$$ | Requires new GPU hardware, application retuning |
| More CPU memory | $$$$ | Limited capacity, still expensive |
| **NVMe SSD** | **$** | **Massive capacity at fraction of cost** |

**Key insight**: If we want to store more KV cache, the traditional approach is to get another machine with more GPU or more CPU memory—both are expensive. **SSD is cheap**. Also, changing GPU hardware means we would have to retune our application based on the new GPU hardware. SSD capacity can be expanded massively with no changes to the application, giving us the **biggest bang for our buck**.

#### Technical Motivation: GPU Memory is Scarce, KV Cache is Big

- **KV-cache dominates memory during decoding**—for modern LLMs, it can consume 60-80% of available GPU memory
- Keeping larger effective caches improves batching efficiency and prefix reuse across requests
- PagedAttention already exploits reuse by splitting KV tensors into small blocks
- We want to **extend that reuse across an SSD tier**

#### Why PagedAttention is Naturally Suited for SSD

Recent work has characterized offloading model weights and KV-cache to NVMe for LLM serving, but **these approaches are not based on paged attention architectures**. They treat KV cache as monolithic structures.

We believe PagedAttention's block-level management is **inherently better suited for SSD-based KV-cache** because:

1. **Block-granularity aligns with I/O operations**: SSDs operate on block-aligned reads/writes. PagedAttention's fixed-size blocks naturally match this.
2. **Maximizes reuse opportunities**: Fine-grained block management allows selective caching of frequently-accessed prefixes
3. **Enables efficient eviction**: Block-level tracking enables LRU/ARC eviction policies that work well with SSD access patterns

### 1.3 Limitations of Current CPU Offloading

The existing vLLM CPU offloading pathway has fundamental limitations:

1. **Data path overhead**: GPU → CPU memory → (future: SSD) requires two memory copies
2. **CPU memory bottleneck**: CPU memory is still limited and expensive
3. **No direct GPU↔SSD path**: Cannot leverage GPUDirect Storage for zero-copy transfers

Our solution addresses these by building an **extensible architecture** that:
- Currently implements CPU offloading as a stepping stone
- Is designed to support **direct GPU↔NVMe transfers via GPUDirect Storage**
- Eliminates CPU memory as a bottleneck in the data path

---

## 2. Overview of Our Solution

### 2.1 Solution Architecture

We designed a **two-tier architecture** that separates scheduling decisions from data transfer execution:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SCHEDULER SIDE                                │
│   (Decides WHAT and WHEN to offload)                                   │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                  OffloadingConnectorScheduler                    │  │
│   │                                                                  │  │
│   │   ┌──────────────────────────────────────────────────────────┐  │  │
│   │   │              OffloadingManager (Abstract)                 │  │  │
│   │   │                                                           │  │  │
│   │   │   ┌─────────────────┐     ┌─────────────────────────┐    │  │  │
│   │   │   │  LRUOffloading  │     │   ARCOffloading         │    │  │  │
│   │   │   │     Manager     │     │      Manager            │    │  │  │
│   │   │   │   (Simple LRU)  │     │  (Adaptive Recency/     │    │  │  │
│   │   │   │                 │     │   Frequency Balancing)  │    │  │  │
│   │   │   └────────┬────────┘     └───────────┬─────────────┘    │  │  │
│   │   │            │                          │                   │  │  │
│   │   │            └────────────┬─────────────┘                   │  │  │
│   │   │                         │                                 │  │  │
│   │   │                  ┌──────▼──────┐                          │  │  │
│   │   │                  │   Backend   │ ◄── Pluggable!           │  │  │
│   │   │                  │  Interface  │     (CPU now, SSD later) │  │  │
│   │   │                  └─────────────┘                          │  │  │
│   │   └──────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                  │                                      │
│                        KVConnectorMetadata                             │
│                     (Transfer specifications)                          │
│                                  │                                      │
└──────────────────────────────────┼──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            WORKER SIDE                                  │
│   (Executes HOW to transfer data)                                      │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                  OffloadingConnectorWorker                       │  │
│   │                                                                  │  │
│   │   ┌──────────────────────────────────────────────────────────┐  │  │
│   │   │                   OffloadingWorker                        │  │  │
│   │   │                                                           │  │  │
│   │   │   ┌───────────────────────────────────────────────────┐  │  │  │
│   │   │   │            OffloadingHandler (Abstract)            │  │  │  │
│   │   │   │                                                    │  │  │  │
│   │   │   │  ┌──────────────────┐  ┌────────────────────────┐ │  │  │  │
│   │   │   │  │ CpuGpuOffloading │  │  [Future: GDSHandler]  │ │  │  │  │
│   │   │   │  │     Handler      │  │   (GPUDirect Storage)  │ │  │  │  │
│   │   │   │  │                  │  │                        │ │  │  │  │
│   │   │   │  │ • d2h_stream     │  │  • Direct GPU↔NVMe     │ │  │  │  │
│   │   │   │  │ • h2d_stream     │  │  • Zero-copy transfers │ │  │  │  │
│   │   │   │  │ • CUDA events    │  │  • Bypass CPU memory   │ │  │  │  │
│   │   │   │  └──────────────────┘  └────────────────────────┘ │  │  │  │
│   │   │   └───────────────────────────────────────────────────┘  │  │  │
│   │   └──────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why This Structure?

We chose this architecture for several key reasons:

1. **Separation of Policy and Mechanism**
   - Scheduler decides *what* to offload (policy)
   - Worker handles *how* to transfer (mechanism)
   - This mirrors vLLM's existing design and enables independent evolution

2. **Pluggable Backend System**
   - Abstract `Backend` interface allows swapping storage mediums
   - CPU backend implemented first as proof-of-concept
   - **Same architecture supports future SSD backend with GPUDirect Storage**
   - No changes needed to scheduler logic when adding new backends

3. **Asynchronous Transfers**
   - GPU memory is **not freed until save completes** (prevents data loss)
   - Requests waiting for cache load **won't be scheduled until load completes**
   - Dedicated CUDA streams prevent interference with model inference

4. **Unified Event System**
   - Allows pulling of cache events (insert, evict, access)
   - Same event stream works for GPU cache and offloaded cache
   - Enables monitoring, metrics, and debugging

### 2.3 Key Components We Implemented

#### 2.3.1 New Offloading System Core

We introduced an abstract `OffloadingManager` interface for managing KV data offloading:

```python
class OffloadingManager(ABC):
    def lookup(self, block_hashes) -> int
        # Find how many consecutive blocks are offloaded
    
    def prepare_load(self, block_hashes) -> LoadStoreSpec
        # Prepare blocks for reading (protect from eviction)
    
    def touch(self, block_hashes)
        # Mark blocks as recently used (for LRU tracking)
    
    def prepare_store(self, block_hashes) -> PrepareStoreOutput
        # Allocate space, possibly evicting old blocks
    
    def complete_store(self, block_hashes, success)
        # Mark blocks as loadable after successful write
    
    def take_events() -> Iterable[OffloadingEvent]
        # Return cache events for monitoring
```

#### 2.3.2 LRU-Based Management

We implemented `LRUOffloadingManager`, a concrete manager using Least Recently Used eviction:

```python
class LRUOffloadingManager(OffloadingManager):
    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend = backend
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
```

Key features:
- `OrderedDict` efficiently tracks access order
- Evicts blocks with `ref_cnt == 0` from front of queue
- `touch()` moves blocks to end (most recently used)

#### 2.3.3 ARC (Adaptive Replacement Cache) Management

We also implemented `ARCOffloadingManager` for workloads that benefit from balancing recency vs. frequency:

```python
class ARCOffloadingManager(OffloadingManager):
    def __init__(self, backend: Backend):
        self.t1: OrderedDict  # Recent cache (accessed once)
        self.t2: OrderedDict  # Frequent cache (accessed multiple times)
        self.b1: OrderedDict  # Ghost list for T1 evictions
        self.b2: OrderedDict  # Ghost list for T2 evictions
        self.target_t1_size: float  # Adaptive target
```

ARC self-tunes between recency and frequency:
- **B1 hit** (recently evicted from T1): Increase `target_t1_size` → favor recency
- **B2 hit** (recently evicted from T2): Decrease `target_t1_size` → favor frequency

#### 2.3.4 CPU Offloading Backend

We implemented a concrete `CPUBackend` for the offloading manager:

```python
class CPUBackend(Backend):
    def __init__(self, block_size: int, num_blocks: int):
        self.num_blocks = num_blocks
        self.allocated_blocks_free_list: list[int] = []
    
    def allocate_blocks(self, block_hashes) -> list[BlockStatus]
    def free(self, block: BlockStatus)
    def get_num_free_blocks() -> int
```

This serves as the **stepping stone toward SSD backend**—the same interface will be used for GPUDirect Storage.

#### 2.3.5 Storage Medium Specifications

We defined `LoadStoreSpec` implementations for different storage mediums:

```python
class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str: return "GPU"

class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str: return "CPU"

# Future: SSDLoadStoreSpec for NVMe
```

#### 2.3.6 Asynchronous Transfer Handler

The `CpuGpuOffloadingHandler` executes actual data transfers:

```python
class CpuGpuOffloadingHandler(OffloadingHandler):
    def __init__(self, ...):
        self.d2h_stream = torch.cuda.Stream()  # GPU → CPU
        self.h2d_stream = torch.cuda.Stream()  # CPU → GPU
        self.transfer_events: dict[int, torch.Event] = {}
    
    def transfer_async(self, job_id, spec) -> bool:
        # Non-blocking transfer using CUDA streams
        with torch.cuda.stream(stream):
            ops.swap_blocks(src, dst, block_mapping)
            event.record(stream)
    
    def get_finished(self) -> list[TransferResult]:
        # Poll CUDA events for completion
```

### 2.4 Changes to vLLM Core

To enable this design, we needed 3 changes in vLLM's core:

1. **Aggregate finished requests on scheduler**: Introduce connector-metadata in the direction worker→scheduler (previously only scheduler→worker was supported)

2. **Move block_hashes to Request**: Introduced `Request.block_hashes` to allow the OffloadingConnector to re-use block hashes computed by KVCacheManager

3. **Support KV events from connectors**: Added connector API for collecting KV cache events (insertion, deletion) to enable unified monitoring

### 2.5 Data Flow

#### Storing KV Blocks (GPU → Offload Storage)

```
1. Scheduler identifies completed blocks ready for offloading
2. OffloadingManager.prepare_store() allocates space, may evict old blocks
3. Scheduler creates OffloadingConnectorMetadata with store specs
4. Worker receives metadata, calls handler.transfer_async()
5. Transfer executes async on dedicated CUDA stream
6. On completion: OffloadingManager.complete_store() marks blocks loadable
```

#### Loading KV Blocks (Offload Storage → GPU)

```
1. Scheduler calls OffloadingManager.lookup() to find offloaded blocks
2. If hit: prepare_load() prepares blocks (prevents eviction during load)
3. Scheduler allocates GPU blocks, creates load specification
4. Worker executes async transfer to GPU
5. On completion: complete_load() releases eviction protection
```

### 2.6 Challenges Encountered

1. **Block Size Alignment**: Offloaded block size can differ from GPU block size. We handle this with `block_size_factor` conversions—one offloaded block may contain multiple GPU blocks.

2. **Reference Counting for Concurrent Access**: Blocks being loaded/stored must not be evicted. We implemented `ref_cnt` in `BlockStatus` to track in-flight transfers.

3. **Asynchronous Completion Tracking**: Used CUDA events with job IDs to track transfer completion without blocking inference.

4. **Cross-Layer KV Cache Support**: Some models share KV cache across layers. Our handler detects this via shape analysis and handles both per-layer and cross-layer caches.

5. **Integration with Existing Scheduler**: Required careful coordination with vLLM's existing `KVCacheManager` and connector infrastructure.

### 2.7 Alternatives Considered (and Why We Discarded Them)

| Alternative | Why Discarded |
|-------------|---------------|
| **Synchronous offloading** | Would stall inference during transfers |
| **Monolithic offload manager** | Not extensible to different backends (CPU/SSD) |
| **FIFO eviction** | LRU/ARC provide significantly better hit rates |
| **Single transfer stream** | Separate d2h/h2d streams allow parallel bidirectional transfers |
| **Implement SSD directly first** | CPU backend simpler to validate architecture correctness |

### 2.8 Testing and CI Integration

We included comprehensive testing at multiple levels:

**Unit Tests** (`test_cpu_manager.py`):
- LRU eviction correctness
- ARC adaptive behavior (T1↔T2 promotion, ghost list adaptation)
- Reference counting
- Event emission

**Integration Tests** (`test_offloading_connector.py`):
- End-to-end scheduler-worker flow
- Block hash verification
- Transfer correctness with mock handlers

**System Tests** (`test_cpu_offloading.py`):
- Full inference with offloading enabled
- Latency verification (CPU cache hit faster than cold miss)
- Output accuracy verification

**CI Integration** (`.buildkite/test-pipeline.yaml`):
```yaml
- pytest -v -s v1/kv_offload
```

---

## 3. Results and Evaluation

### 3.1 Latency Comparison

Our system test verifies that CPU cache hits are faster than cold misses:

```python
def _latency_test(llm, subscriber):
    # Cold miss: Full computation
    # GPU hit: Prefix cache hit (fastest)
    # CPU hit: Load from CPU offload (faster than cold)
    
    assert num_times_cpu_better_than_cold >= 0.8 * num_tests
```

### 3.2 Accuracy Verification

We verify that offloading doesn't affect model outputs:

```python
def _accuracy_test(llm, subscriber):
    prompt = "Let's count to 10. One, two, three, four,"
    # Verify model correctly outputs " five" after loading from CPU cache
    assert success_count >= 0.5 * test_count
```

---

## 4. Future Work: Path to Full SSD Support

Our architecture is designed to support **GPUDirect Storage** for direct GPU↔NVMe transfers:

### 4.1 What's Needed for SSD Backend

1. **New SSDBackend class** implementing the `Backend` interface
2. **GDSOffloadingHandler** using NVIDIA GPUDirect Storage APIs
3. **SSDLoadStoreSpec** with file offsets/addresses

### 4.2 Expected Benefits of SSD

| Metric | CPU Offloading | SSD with GDS |
|--------|----------------|--------------|
| Capacity | ~100GB | **Multiple TB** |
| Cost per GB | $$$ | **$** |
| Data path | GPU↔CPU↔(Storage) | **GPU↔SSD direct** |
| CPU overhead | Memory copies | **Zero-copy** |

### 4.3 Additional Future Enhancements

- **Tiered offloading**: GPU → CPU → SSD hierarchy
- **Compression**: Compress KV data before offloading
- **Prefetching**: Predict and preload likely-needed blocks
- **Distributed offloading**: Share cache across nodes

---

## 5. Usage

### 5.1 Configuration

```python
from vllm import LLM
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 1000,      # Number of CPU blocks
        "block_size": 48,            # Offloaded block size
        "eviction_policy": "lru",    # or "arc"
    },
)

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
)
```

### 5.2 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `num_cpu_blocks` | Blocks to allocate in CPU memory | Required |
| `block_size` | Offloaded block size (tokens) | GPU block size |
| `eviction_policy` | `lru` or `arc` | `lru` |
| `spec_name` | OffloadingSpec implementation | `CPUOffloadingSpec` |

---

## 6. Conclusion

We successfully implemented an **extensible KV cache offloading system** for vLLM that:

✅ **Enables efficient offloading** with asynchronous GPU↔CPU transfers  
✅ **Implements intelligent eviction** with both LRU and ARC policies  
✅ **Provides pluggable backend architecture** designed for future SSD support  
✅ **Integrates seamlessly** with vLLM's PagedAttention system  
✅ **Includes comprehensive testing** at unit, integration, and system levels  

The architecture positions vLLM to support **SSD-backed KV caching with GPUDirect Storage**, enabling massive KV cache capacity expansion at minimal cost—the **biggest bang for the buck** in scaling LLM serving systems.

---

## References

1. vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
2. NVIDIA GPUDirect Storage Documentation
3. ARC: A Self-Tuning, Low Overhead Replacement Cache (Megiddo & Modha, 2003)
