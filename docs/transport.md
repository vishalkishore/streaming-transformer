# Host/GPU transport and synchronization

The low-latency path uses `cudaHostAllocMapped` storage and its CUDA device alias. Request, completion and exited sequence words are 64-bit integers, each on a separate 64-byte alignment boundary. Input and output storage are separate. The selected platform is Linux x86-64 with mapped-host-memory and unified-addressing support, checked by the hardware probe.

Both host and device use `cuda::atomic_ref<uint64_t, cuda::thread_scope_system>` loads/stores. There is no cross-CPU/GPU atomic read-modify-write operation. The documented CCCL model supports naturally aligned mapped-memory loads/stores of this size even when `hostNativeAtomicSupported` is zero; that flag therefore does not authorize RMW operations here. See NVIDIA's [memory model and mapped-memory atomicity rules](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html). The installed CUDA toolkit supplies the actual host/device atomic implementation.

## Request and completion

1. The CPU fills the pinned input and command, then stores the next request ID with release ordering.
2. The GPU coordinator loads the ID with acquire ordering. A grid/block barrier publishes its chosen command to the other GPU threads.
3. GPU threads stage the event and execute the forward pass. Phase barriers connect each operation's output owners to its consumers. The final barrier completes device state and output writes.
4. The coordinator writes the mapped score and stores the completion ID with release ordering. The CPU acquires that exact ID before reading the score or reusing the input.

The CPU never treats a numerical score as a completion sentinel. Only one request is outstanding, so buffers are not overwritten while GPU readers still use them. Correctness snapshots occur after completion and before a subsequent command is published.

RUN advances model position once; RESET clears state without consuming a RUN budget; STOP exits and is followed by stream synchronization before resources are freed. An idle exit publishes the exited flag. The last sequence ID is reserved for STOP so exhaustion cannot strand a live worker. The host refuses further RUN/RESET requests and requires a reconstructed runtime before sequence wraparound.

Tight polling and bounded backoff preserve these orderings. Device backoff uses nanosleep and the host yields while waiting. They change responsiveness and resource use, not publication semantics. A successful stress test or sanitizer run is supporting evidence, not a proof of the cross-CPU/GPU memory model.

## Ordered-copy and graph paths

The correctness baseline copies pinned CPU input to preallocated device memory, launches the conventional/graph forward, copies the score to pinned output, and synchronizes the stream. Mapped graph staging instead reads/writes fixed pinned aliases in the captured sequence and still synchronizes for host visibility. Transport differences are included in host-service latency and recorded in result metadata.

## Transport measurements

`ping-pong` is CPU initiated: publish input/request, GPU validates payload/checksum and acknowledges, CPU observes completion. `gpu-ping` is GPU initiated: GPU publishes a sequence/checksum request, CPU observes and acknowledges it, then GPU checks the acknowledgment. GPU-initiated device round trips and CPU-initiated device wait intervals are saved in device-cycle samples; the host observation samples remain a separate clock-domain measurement.

The direction experiments have different payload/acknowledgment work and must be interpreted with their recorded boundaries. An expensive remote request read or cheap posted completion write is a hypothesis to test, not a universal PCIe rule. A GPU store timestamp does not establish when the CPU observed it. No result divides an RTT in half or subtracts unrelated host/GPU timestamps to claim a one-way leg time.

Empty launch, empty graph and pinned-copy/echo modes provide additional baselines. Each exchange checks IDs or payloads; raw samples, polling policy, CPU time, hardware and executable/source provenance are preserved. Final repetitions and before/after controls must be inspected separately from earlier shared-GPU smoke results.
