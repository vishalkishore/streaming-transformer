"""Record actual hardware/software state without changing clocks or other jobs."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import re
from pathlib import Path
import torch


def capture(command):
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, error=str(error))


def collect(probe='build-blackwell/stream-probe', nvcc=None, binary='build-blackwell/stream-bench'):
    nvcc = nvcc or shutil.which('nvcc') or 'nvcc'
    linkage = capture(['ldd', binary])
    libraries = []
    for path in re.findall(r'=> (/\S+)', linkage.get('stdout', '')):
        resolved = Path(path).resolve()
        libraries.append(dict(path=path, resolved=str(resolved), bytes=resolved.stat().st_size))
    return dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                torch_cuda=torch.version.cuda, cpu_affinity=sorted(os.sched_getaffinity(0)),
                torch_config=torch.__config__.show(), native_linkage=linkage, native_libraries=libraries,
                commands=[capture(c) for c in [
                    [probe], ['lscpu', '-J'], ['lscpu', '-e'], ['numactl', '--hardware'],
                    ['nvidia-smi', '-q'], ['nvidia-smi', 'topo', '-m'],
                    [nvcc, '--version'], ['g++', '--version'], ['uv', '--version'],
                    ['uv', 'pip', 'freeze', '--python', os.sys.executable],
                    ['cmake', '--version'], ['git', 'rev-parse', 'HEAD'], ['git', 'status', '--porcelain'],
                    ['ninja', '-C', str(Path(binary).parent), '-t', 'commands', Path(binary).name],
                    ['cat', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'],
                    ['cat', '/sys/devices/system/cpu/cpufreq/boost']]])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/environment.json')
    p.add_argument('--probe', default='build-blackwell/stream-probe')
    p.add_argument('--nvcc')
    p.add_argument('--binary', default='build-blackwell/stream-bench')
    a = p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(collect(a.probe, a.nvcc, a.binary), indent=2))
