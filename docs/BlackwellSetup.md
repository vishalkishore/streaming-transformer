# Blackwell setup and fixture regeneration

The final target selected on 27 October 2026 is the NVIDIA RTX PRO 5000 Blackwell, 48 GB, SM120. The recorded device has 110 SMs. Historical A5000 results and the original A6000 brief are not final Blackwell evidence.

Use `uv` for Python dependency installation and removal. The oracle environment
retains PyTorch 2.6.0 and executes on the CPU to reproduce transferred fixtures.
The current installed distribution reports `2.6.0+cu124`; its CUDA backend is not
used for Blackwell inference. A separate `.venv-gpu` contains Torch 2.14.1+cu130
and Triton 3.8.0, with its installed package set in
`bench/requirements-blackwell.lock`. Its eager/compile/explicit-graph benchmark
is implemented; the final 18-condition CUDA correctness matrix remains pending.

From the repository root, with CUDA 13.0, a supported compiler and the native development dependencies already available:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest matplotlib
uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync python -m scripts.fixtures
uv run --no-sync python -m scripts.fixtures --verify-only
cmake -S . -B build-blackwell -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-blackwell -j 6
build-blackwell/stream-probe results/blackwell-hardware.json
uv run --no-sync python scripts/environment.py --probe build-blackwell/stream-probe --output results/blackwell-environment.json
uv run --no-sync python -m pytest -q
ctest --test-dir build-blackwell --output-on-failure
uv run --no-sync python scripts/validate.py --binary build-blackwell/stream-bench --output results/blackwell-validation
uv run --no-sync python -m scripts.diagnose_native
```

`--no-sync` uses the explicitly installed oracle environment without resolving a different Torch distribution. Add/remove dependencies with `uv pip install` / `uv pip uninstall --python .venv/bin/python`, or update project dependencies with `uv add` / `uv remove` when changing the declared environment.

The default fixture command regenerates Small/Main in both modes, using seed 42 weights, seed 123 traces and the original reset schedule. Approximately 794 MiB of fixtures are produced, including complete per-event KV snapshots. Hash verification checks the checkpoint, expected archive, individual tensors, input trace and reset schedule. Large fixtures are opt-in with `--sizes large`; their full KV histories are substantially larger. Regeneration is deterministic for the recorded oracle environment, and hashes must be compared when changing Torch versions.

CMake discovers the toolkit and host compiler; explicit `CMAKE_CUDA_COMPILER` / `CMAKE_CUDA_HOST_COMPILER` overrides remain available. Use a fresh build directory when switching toolkits. SM86 remains an explicit build option for the previous hardware.

The fresh all-size native matrix passed 54/54 conditions under the documented
`mixed-depth-rms-v3` policy. Original scalar/head failures remain diagnostics.
The validation runner skips stress whenever fixture correctness fails. The four
million-command cases and additional adversarial/window/PyTorch checks must pass
before final performance studies. See the active handoff for live job details.

PyTorch compile explicitly preserves FP16 cast boundaries with
`emulate_precision_casts=True`; TF32, FP16 accumulation, reduced-precision FP16
reductions, automatic compile graph capture and fast math are disabled. Run the
CPU-only compiler witness with `.venv-gpu/bin/python -m
scripts.check_torch_casts`; it hides GPUs before importing Torch. Its passing
result is separate from complete CUDA model correctness.
