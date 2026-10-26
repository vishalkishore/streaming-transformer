"""Run remaining correctness checks serially after the native stress gate."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from bench.gates import check_native_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--gate', type=Path, default=Path('results/acceptance-release/summary.json'))
    parser.add_argument('--output', type=Path, default=Path('results/validation-followup'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        process = Path(f'/proc/{args.wait_pid}')
        if process.exists():
            command = (process / 'cmdline').read_bytes()
            if b'scripts/validate.py' not in command or str(args.gate.parent).encode() not in command:
                raise RuntimeError('Wait PID does not identify the intended validation job')
            # The start time distinguishes this process from a subsequently reused PID.
            identity = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
            print(f'Waiting for validation PID {args.wait_pid}', flush=True)
            while process.exists():
                try:
                    fields = (process / 'stat').read_text().split(') ', 1)[1].split()
                except FileNotFoundError:
                    break
                if fields[19] != identity or fields[0] == 'Z':
                    break
                time.sleep(5)
    binary = Path('build-blackwell/stream-bench')
    check_native_gate(args.gate, hashlib.sha256(binary.read_bytes()).hexdigest())
    print('Complete native correctness/stress gate accepted', flush=True)
    commands = [
        ('ctest', ['ctest', '--test-dir', 'build-blackwell', '--output-on-failure']),
        ('memcheck', ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                      'build-blackwell/stream-tests']),
        ('synccheck', ['compute-sanitizer', '--tool', 'synccheck', '--error-exitcode', '99',
                      'build-blackwell/stream-tests']),
        ('extended', [sys.executable, '-m', 'scripts.extended_validation']),
        ('windows', [sys.executable, '-m', 'scripts.check_windows']),
        ('torch', [sys.executable, '-m', 'scripts.validate_torch']),
    ]
    report = []
    for name, command in commands:
        clients = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip()
        if clients:
            raise RuntimeError(f'Another GPU client is active; no job was interrupted: {clients}')
        print(f'Starting {name}', flush=True)
        with (args.output / f'{name}.log').open('w') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        report.append(dict(name=name, command=command, returncode=rc))
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
        if rc:
            raise RuntimeError(f'{name} failed; inspect {args.output / (name + ".log")}')
        print(f'{name} passed', flush=True)


if __name__ == '__main__':
    main()
