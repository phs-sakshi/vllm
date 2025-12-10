# SSD-Backed KV Cache PagedAttention for vLLM

## Project Report

---

## 1. Problem Statement and Motivation

### 1.1 The Problem

PagedAttention, a key component of modern Large Language Model (LLM) serving systems like vLLM, stores key-value (KV) tensors in non-contiguous, fixed-size blocks ("pages") to achieve high throughput by reducing memory fragmentation and enabling fine-grained reuse across requests. However, the existing system lacks a fast "swap" path to secondary storage—when GPU memory fills up, blocks are reclaimed and future reuse opportunities are permanently lost.

The current vLLM implementation only supports offloading excess KV cache to CPU memory. While this provides some relief, it comes with significant limitations:

1. **Cost**: Both GPU memory and CPU memory are expensive resources
2. **Capacity**: CPU memory, while larger than GPU memory, is still limited
3. **Scalability**: Scaling up requires procuring machines with more GPUs or more CPU memory, both expensive propositions that require application retuning

### 1.2 Why This Problem Matters

**GPU Memory Scarcity vs. KV Cache Size**: KV-cache dominates memory during the decoding phase of LLM inference. For modern large language models, the KV cache can consume the majority of available GPU memory. Keeping larger effective caches improves batching efficiency and prefix reuse across requests.

**Economic Considerations**: 
- Adding more GPU memory means acquiring new hardware, which is extremely expensive
- Additional GPU hardware requires retuning the application for the new hardware characteristics
- SSDs provide massive storage capacity at a fraction of the cost per GB compared to GPU/CPU memory
- SSD capacity can be expanded with minimal application changes, providing the "biggest bang for the buck"

**PagedAttention's Natural Fit for SSD Offloading**: PagedAttention already exploits reuse by splitting KV tensors into small blocks. This block-level management is inherently well-suited for SSD-based caching, as it naturally aligns with block-granularity I/O operations and maximizes reuse opportunities.

### 1.3 Why Existing Solutions Fall Short

Recent research has characterized offloading model weights and KV-cache to NVMe for LLM serving, but these approaches are not based on paged attention architectures. They treat KV cache as monolithic structures rather than leveraging the page-level granularity that PagedAttention provides.

The current vLLM implementation's CPU-only offloading pathway:
- Uses synchronous transfers that can stall inference
- Has limited capacity bounded by available CPU memory
- Does not exploit the block-level management that PagedAttention enables

### 1.4 Our Proposal

We propose to add an SSD storage tier for KV blocks and enable seamless data movement. Our implementation creates an extensible offloading framework that:
- Enables blocks of KV cache to be evicted to and fetched from secondary storage
- Uses asynchronous transfers to avoid stalls
- Implements intelligent eviction policies (LRU and ARC) to maximize cache efficiency
- Provides a pluggable backend system that can be extended to support NVMe SSDs with GPUDirect Storage, enabling direct GPU↔SSD transfers that bypass system memory entirely

---

## 2. Overview of the Solution

### 2.1 Architecture Overview

Our solution extends vLLM's v1 engine with a comprehensive KV cache offloading system. The architecture follows a clean separation between scheduler-side logic and worker-side execution.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCHEDULER SIDE                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OffloadingConnectorScheduler                │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │           OffloadingManager (Abstract)          │    │   │
│  │  │  ┌─────────────────┐  ┌──────────────────────┐ │    │   │
│  │  │  │ LRUOffloading   │  │  ARCOffloading       │ │    │   │
│  │  │  │    Manager      │  │     Manager          │ │    │   │
│  │  │  └────────┬────────┘  └──────────┬───────────┘ │    │   │
│  │  │           │                      │             │    │   │
│  │  │           └──────────┬───────────┘             │    │   │
│  │  │                      │                         │    │   │
│  │  │              ┌───────▼───────┐                 │    │   │
│  │  │              │ Backend (CPU) │                 │    │   │
│  │  │              └───────────────┘                 │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                    KVConnectorMetadata                         │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         WORKER SIDE                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               OffloadingConnectorWorker                  │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              OffloadingWorker                   │    │   │
│  │  │  ┌─────────────────────────────────────────┐   │    │   │
│  │  │  │         OffloadingHandler               │   │    │   │
│  │  │  │    (e.g., CpuGpuOffloadingHandler)      │   │    │   │
│  │  │  │  ┌───────────────┐ ┌─────────────────┐  │   │    │   │
│  │  │  │  │ CUDA Streams  │ │  Event Pool     │  │   │    │   │
│  │  │  │  │ (d2h, h2d)    │ │                 │  │   │    │   │
│  │  │  │  └───────────────┘ └─────────────────┘  │   │    │   │
│  │  │  └─────────────────────────────────────────┘   │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

#### 2.2.1 OffloadingManager (Abstract Interface)

The `OffloadingManager` is an abstract class that defines the interface for managing KV data offloading:

```python
class OffloadingManager(ABC):
    def lookup(self, block_hashes: Iterable[BlockHash]) -> int
    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec
    def touch(self, block_hashes: Iterable[BlockHash])
    def complete_load(self, block_hashes: Iterable[BlockHash])
    def prepare_store(self, block_hashes: Iterable[BlockHash]) -> PrepareStoreOutput | None
    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool)
    def take_events(self) -> Iterable[OffloadingEvent]
```

Key primitives:
- **lookup()**: Finds the length of the maximal series of blocks that are all offloaded
- **prepare_load()**: Prepares blocks to be read (protects from eviction)
- **touch()**: Marks blocks as recently used for LRU tracking
- **complete_load()**: Re-allows eviction of loaded blocks
- **prepare_store()**: Prepares blocks for writing (may trigger evictions)
- **complete_store()**: Marks blocks as loadable after successful write
- **take_events()**: Returns cache events for external monitoring

#### 2.2.2 LRUOffloadingManager

Implements the Least Recently Used (LRU) eviction policy:

```python
class LRUOffloadingManager(OffloadingManager):
    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
```

Key features:
- Uses `OrderedDict` to efficiently track access order
- Evicts blocks with `ref_cnt == 0` from the front of the queue
- Moves accessed blocks to the end via `touch()`
- Supports event emission for external monitoring

#### 2.2.3 ARCOffloadingManager

Implements the Adaptive Replacement Cache (ARC) eviction policy, which self-tunes between recency and frequency:

```python
class ARCOffloadingManager(OffloadingManager):
    def __init__(self, backend: Backend, enable_events: bool = False):
        self.backend: Backend = backend
        self.target_t1_size: float = 0.0
        self.t1: OrderedDict[BlockHash, BlockStatus] = OrderedDict()  # Recent
        self.t2: OrderedDict[BlockHash, BlockStatus] = OrderedDict()  # Frequent
        self.b1: OrderedDict[BlockHash, None] = OrderedDict()  # Ghost list T1
        self.b2: OrderedDict[BlockHash, None] = OrderedDict()  # Ghost list T2
```

ARC Algorithm:
1. **T1 (Recent)**: Contains blocks accessed once
2. **T2 (Frequent)**: Contains blocks accessed multiple times
3. **B1/B2 (Ghost Lists)**: Track recently evicted blocks to adapt behavior
4. **Adaptive Target**: 
   - B1 hit → increase `target_t1_size` (favor recency)
   - B2 hit → decrease `target_t1_size` (favor frequency)

#### 2.2.4 Backend Interface

Abstract interface for storage backends:

```python
class Backend(ABC):
    def __init__(self, block_size: int, medium: str)
    def get_num_free_blocks(self) -> int
    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]
    def free(self, block: BlockStatus)
    def get_load_store_spec(...) -> LoadStoreSpec
```

#### 2.2.5 CPUBackend

Concrete implementation for CPU memory offloading:

```python
class CPUBackend(Backend):
    def __init__(self, block_size: int, num_blocks: int):
        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []
```

Features:
- Manages a fixed pool of CPU memory blocks
- Reuses freed blocks via a free list
- Returns `CPULoadStoreSpec` with block IDs for transfer operations

#### 2.2.6 LoadStoreSpec Hierarchy

Specifications for different storage mediums:

```python
class LoadStoreSpec(ABC):
    @staticmethod
    @abstractmethod
    def medium() -> str: pass

class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str: return "GPU"

class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str: return "CPU"
```

#### 2.2.7 OffloadingWorker and Handlers

Worker-side transfer execution:

```python
class OffloadingWorker:
    def __init__(self):
        self.handlers: set[OffloadingHandler] = set()
        self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}

    def register_handler(self, src_cls, dst_cls, handler)
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool
    def get_finished(self) -> list[TransferResult]
```

The `CpuGpuOffloadingHandler` handles CPU↔GPU transfers:

```python
class CpuGpuOffloadingHandler(OffloadingHandler):
    def __init__(self, ...):
        self.d2h_stream = torch.cuda.Stream()  # GPU→CPU
        self.h2d_stream = torch.cuda.Stream()  # CPU→GPU
        self.transfer_events: dict[int, torch.Event] = {}
        self.events_pool: list[torch.Event] = []
```

Key features:
- Separate CUDA streams for each transfer direction
- Non-blocking transfers using CUDA events
- Event pooling to reduce allocation overhead
- Uses vLLM's `swap_blocks` operation for efficient data movement

#### 2.2.8 OffloadingConnector

Main connector implementing vLLM's `KVConnectorBase_V1`:

```python
class OffloadingConnector(KVConnectorBase_V1):
    prefer_cross_layer_blocks: ClassVar[bool] = True

    def __init__(self, vllm_config, role, kv_cache_config):
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)
```

### 2.3 Data Flow

#### 2.3.1 Storing KV Blocks (GPU → Offload)

1. Scheduler identifies completed blocks ready for offloading
2. `OffloadingManager.prepare_store()` allocates offload space and may evict old blocks
3. Scheduler creates `OffloadingConnectorMetadata` with store specifications
4. Worker receives metadata and calls `OffloadingHandler.transfer_async()`
5. Transfer executes asynchronously on dedicated CUDA stream
6. On completion, `OffloadingManager.complete_store()` marks blocks as loadable

#### 2.3.2 Loading KV Blocks (Offload → GPU)

1. Scheduler calls `OffloadingManager.lookup()` to find offloaded blocks
2. If hit, `OffloadingManager.prepare_load()` prepares blocks (prevents eviction)
3. Scheduler allocates GPU blocks and creates load specification
4. Worker executes async transfer from offload medium to GPU
5. On completion, `OffloadingManager.complete_load()` releases eviction protection

### 2.4 Why This Architecture?

**Separation of Concerns**: The scheduler handles high-level decisions (what to offload, when to load) while workers handle actual data movement. This mirrors vLLM's existing architecture.

**Asynchronous Transfers**: 
- GPU memory is not freed until save completes (no data loss risk)
- Requests waiting for cache load are not scheduled until load completes
- Dedicated CUDA streams prevent interference with inference

**Pluggable Backend System**: The abstract `Backend` interface allows easy extension to:
- NVMe SSDs with GPUDirect Storage
- Remote storage systems
- Tiered storage configurations

**Unified Event System**: The `OffloadingEvent` and `KVCacheEvent` integration allows external systems to monitor cache behavior, enabling:
- Metrics collection
- Cache analytics
- Debugging and profiling

### 2.5 Challenges Encountered

1. **Block Size Alignment**: The offloaded block size can differ from GPU block size. The implementation handles this with `block_size_factor` conversions throughout the codebase.

2. **Reference Counting**: Blocks being loaded or stored must not be evicted. The `ref_cnt` mechanism in `BlockStatus` prevents this race condition.

3. **Asynchronous Completion Tracking**: Job IDs and CUDA events must be carefully managed to track transfer completion without blocking inference.

4. **Cross-Layer KV Cache Support**: Some models use cross-layer KV caches where all layers share a single cache tensor. The handler detects this via shape analysis and handles both cases.

5. **Integration with Scheduler**: Required changes to vLLM core:
   - Aggregating finished requests on the scheduler
   - Moving block hashes from `KVCacheManager` to `Request`
   - Supporting KV events from connectors

### 2.6 Alternatives Considered

1. **Synchronous Offloading**: Rejected due to stalling inference during transfers.

2. **Monolithic Offload Manager**: Rejected in favor of pluggable backends for extensibility.

3. **FIFO Eviction**: Rejected in favor of LRU/ARC which provide better cache hit rates.

4. **Single Transfer Stream**: Rejected; separate d2h/h2d streams allow bidirectional transfers in parallel.

5. **Direct SSD Implementation First**: Chose CPU backend first as a simpler testbed; the architecture supports SSD backends as future work.

### 2.7 Testing and CI Integration

Comprehensive testing at multiple levels:

1. **Unit Tests** (`test_cpu_manager.py`):
   - LRU eviction behavior
   - ARC adaptive behavior
   - Reference counting correctness
   - Event emission

2. **Integration Tests** (`test_offloading_connector.py`):
   - End-to-end scheduler-worker flow
   - Block hash verification
   - Transfer correctness

3. **System Tests** (`test_cpu_offloading.py`):
   - Full inference with offloading enabled
   - Latency verification (CPU hit faster than cold)
   - Accuracy verification (correct outputs)

4. **CI Integration**: Tests run in `.buildkite/test-pipeline.yaml`:
   ```yaml
   - pytest -v -s v1/kv_offload
   ```

---

## 3. Configuration and Usage

### 3.1 Basic Usage

```python
from vllm import LLM
from vllm.config import KVTransferConfig

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": 1000,      # Number of CPU blocks for offloading
        "block_size": 48,            # Offloaded block size (tokens)
        "eviction_policy": "lru",    # or "arc"
    },
)

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.5,
    kv_transfer_config=kv_transfer_config,
)
```

### 3.2 Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `num_cpu_blocks` | Number of blocks to allocate in CPU memory | Required |
| `block_size` | Size of offloaded blocks in tokens | GPU block size |
| `eviction_policy` | Eviction algorithm (`lru` or `arc`) | `lru` |
| `spec_name` | OffloadingSpec implementation | `CPUOffloadingSpec` |

---

## 4. Future Work

1. **SSD Backend with GPUDirect Storage**: Direct GPU↔NVMe transfers bypassing CPU
2. **Tiered Offloading**: CPU cache tier before SSD tier
3. **Compression**: Compress KV data before offloading
4. **Prefetching**: Predict and preload likely-needed blocks
5. **Distributed Offloading**: Share offloaded cache across nodes

---

## 5. Conclusion

This project successfully implements an extensible KV cache offloading system for vLLM that:

- Enables efficient offloading to CPU memory with asynchronous transfers
- Implements both LRU and ARC eviction policies
- Provides a pluggable backend system for future SSD support
- Integrates seamlessly with vLLM's existing PagedAttention architecture
- Includes comprehensive testing at unit, integration, and system levels

The architecture positions vLLM to support SSD-backed KV caching with GPUDirect Storage, enabling massive KV cache capacity expansion at minimal cost.
