"""File and repository identities collected outside measured execution."""
import hashlib
import subprocess
import sys
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def python_provenance():
    def git(*arguments):
        result = subprocess.run(['git', *arguments], text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    sources = sorted(set(Path('bench').glob('*.py')) | set(Path('reference').glob('*.py')))
    sources += [Path('bench/requirements-blackwell.lock')]
    hashes = {str(path): file_hash(path) for path in sources}
    status = git('status', '--porcelain')
    return dict(git_commit=git('rev-parse', '--verify', 'HEAD'), git_status=status,
                git_dirty=status != '', source_sha256=hashes,
                python_version=sys.version, python_executable=str(Path(sys.executable).resolve()),
                python_executable_sha256=file_hash(sys.executable))


def native_libraries(binary):
    output = subprocess.check_output(['ldd', str(binary)], text=True)
    if 'not found' in output:
        raise RuntimeError('Benchmark has unresolved shared libraries')
    libraries = {}
    for line in output.splitlines():
        for field in line.split():
            if field.startswith('/'):
                path = Path(field).resolve()
                libraries[str(path)] = dict(bytes=path.stat().st_size, sha256=file_hash(path))
    return libraries
