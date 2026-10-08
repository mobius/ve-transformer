"""Check validated default selection with temporary files, without hardware runs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from qwen_session import accepted_executor

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    candidate=root/'build/qwen15-input-reuse/qwen-infer-ve'
    projection=root/'build/qwen15-projection/qwen-infer-ve'
    session=root/'build/qwen15-session/qwen-infer-ve'
    report=root/'docs/results/qwen25-1.5b-input-reuse.json'
    for path in (candidate,projection,session):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'fixture')
    report.parent.mkdir(parents=True)
    assert accepted_executor(root)==projection
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    for data in ({'adopted':False,'candidate_binary_sha256':digest}, [],
                 {'adopted':True,'candidate_binary_sha256':'stale'}, {'adopted':True}):
        report.write_text(json.dumps(data))
        assert accepted_executor(root)==projection
    report.write_text('{')
    assert accepted_executor(root)==projection
    report.write_text(json.dumps({'adopted':True,'candidate_binary_sha256':digest}))
    assert accepted_executor(root)==candidate
    candidate.write_bytes(b'rebuilt fixture')
    assert accepted_executor(root)==projection
    projection.unlink()
    assert accepted_executor(root)==session
    print('Validated executor selection: missing, malformed, rejected, accepted, rebuild and fallback PASS')
