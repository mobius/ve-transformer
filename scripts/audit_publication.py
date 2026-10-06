"""Read-only privacy audit of the exact staged Git contents; prints no matches."""
import re
import subprocess
from pathlib import PurePosixPath


def main():
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'])
    patterns = {
        'private_path': re.compile(rb'/ho[m]e/|/mnt/stor[a]ge/'),
        'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        'access_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})'),
        'credential_assignment': re.compile(rb"(?i)(?:password|passwd|secret|api_key|access_token)\s*[:=]\s*[\"'][^\"'\n]{6,}[\"']"),
        'email': re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    }
    findings = []
    count = 0
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, oid, stage = meta.split()
        name = name.decode('utf-8')
        path = PurePosixPath(name)
        count += 1
        if stage != b'0' or mode not in (b'100644', b'100755'):
            findings.append((name, 'unexpected_index_entry'))
            continue
        if path.parts[0] in ('build', '.venv', '.git-local') or path.name.startswith('.env') or path.suffix in ('.pem', '.key', '.p12', '.pfx', '.gguf'):
            findings.append((name, 'local_or_sensitive_artifact'))
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid])
        if len(data) > 1_000_000 or b'\0' in data:
            findings.append((name, 'large_or_binary_file'))
        for label, pattern in patterns.items():
            if pattern.search(data):
                findings.append((name, label))
    print('Staged files audited:', count)
    for name, label in findings:
        print('Review required:', name, label)
    if findings:
        raise SystemExit(1)
    print('PASS: no findings in configured checks; manual review still required.')


if __name__ == '__main__':
    main()
