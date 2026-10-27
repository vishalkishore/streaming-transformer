"""Wait for an unused GPU and reject runs with observed competing clients."""
import fcntl
import json
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def gpu_processes():
    # XML includes both graphics and compute clients; compute-apps alone does not.
    output = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=10)
    root = ET.fromstring(output)
    result = []
    for gpu in root.findall('gpu'):
        processes = gpu.find('processes')
        if processes is None or (processes.text or '').strip() in ('N/A', 'Not Supported'):
            raise RuntimeError('GPU process visibility is unavailable')
        for process in processes.findall('process_info'):
            result.append(dict(gpu=gpu.findtext('uuid'), pid=int(process.findtext('pid')),
                               name=process.findtext('process_name'), type=process.findtext('type')))
    if not root.findall('gpu'):
        raise RuntimeError('No GPU was reported by nvidia-smi')
    return result


def belongs_to_run(pid, root_pid):
    seen = set()
    while pid and pid not in seen:
        if pid == root_pid:
            return True
        seen.add(pid)
        try:
            status = Path(f'/proc/{pid}/status').read_text().splitlines()
            pid = int(next(line.split(':', 1)[1] for line in status if line.startswith('PPid:')))
        except (OSError, StopIteration, ValueError):
            return False
    return False


class GpuGuard:
    def __init__(self, evidence):
        self.evidence = Path(evidence)
        self.stop = threading.Event()
        self.thread = None
        self.pid = None
        self.violations = []
        self.samples = 0
        self.lock = None

    def __enter__(self):
        # Serialize this workspace's launchers as well as checking external jobs.
        Path('results').mkdir(exist_ok=True)
        self.lock = Path('results/.gpu-run.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX)
            reported = None
            while True:
                clients = gpu_processes()
                if not clients:
                    break
                description = json.dumps(clients)
                if description != reported:
                    print(f'Waiting for an empty GPU: {description}', file=sys.stderr, flush=True)
                    reported = description
                time.sleep(5)
            self.start = time.time_ns()
            return self
        except BaseException:
            self.lock.close()
            raise

    def sample(self):
        try:
            clients = gpu_processes()
            self.samples += 1
            foreign = [client for client in clients if not belongs_to_run(client['pid'], self.pid)]
            if foreign:
                self.violations.append(dict(time_ns=time.time_ns(), clients=foreign))
        except Exception as error:
            self.violations.append(dict(time_ns=time.time_ns(), query_error=str(error)))

    def watch(self, pid):
        self.pid = pid

        def monitor():
            while not self.stop.is_set():
                self.sample()
                self.stop.wait(1)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def __exit__(self, kind, value, traceback):
        try:
            self.stop.set()
            if self.thread:
                self.thread.join()
            self.sample()
            self.evidence.parent.mkdir(parents=True, exist_ok=True)
            self.evidence.write_text(json.dumps(dict(
                empty_before_launch=True, child_pid=self.pid, start_ns=self.start,
                end_ns=time.time_ns(), samples=self.samples, interval_seconds=1,
                valid=not self.violations, violations=self.violations,
                limitation='Sampled process visibility cannot detect every short-lived competing process.'), indent=2))
        finally:
            self.lock.close()
        if kind is None and self.violations:
            raise RuntimeError(f'GPU run invalid: competing process or visibility failure; see {self.evidence}')


def run_exclusive(command, *, evidence, check=False, **kwargs):
    with GpuGuard(evidence) as guard:
        with subprocess.Popen(command, **kwargs) as child:
            guard.watch(child.pid)
            stdout, stderr = child.communicate()
            result = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, stdout, stderr)
    return result
