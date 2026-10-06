"""Serial, stateless requests to a resident native Qwen executor (no network server)."""
import json
import queue
import struct
import subprocess
import threading
import time


def chat(prompt):
    return '<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'


class QwenSession:
    def __init__(self, command, stderr, env=None, timeout=300):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.lines = queue.Queue()
        started = time.perf_counter()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=stderr, env=env)
        def reader():
            try:
                for line in self.process.stdout:
                    self.lines.put(line)
            finally:
                self.lines.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            self.ready = self._read()
            if self.ready.get('event') != 'ready' or self.ready.get('protocol') != 1:
                raise RuntimeError('unexpected session handshake')
            self.startup_seconds = time.perf_counter()-started
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError('native session response timed out') from None
        if line is None:
            raise RuntimeError('native session closed; inspect local stderr log')
        return json.loads(line)

    def request(self, prompt, count=32, raw=False):
        payload = (prompt if raw else chat(prompt)).encode('utf-8')
        if len(payload) > 1024*1024 or not 1 <= count <= 1024:
            raise ValueError('request exceeds protocol bounds')
        with self.lock:
            started = time.perf_counter()
            try:
                self.process.stdin.write(struct.pack('<II', len(payload), count)+payload)
                self.process.stdin.flush()
                row = self._read()
            except BaseException:
                self.close()
                raise
            row['host_roundtrip_seconds'] = time.perf_counter()-started
            return row

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                code = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                code = self.process.wait()
        self.reader.join(timeout=2)
        self.process.stdout.close()
        return code


def main():
    import argparse
    import os
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Sequential resident VE requests; launch under temperature_guard.py')
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--slot', type=int, default=1)
    parser.add_argument('--stderr-log', default='build/qwen15-session/client.stderr.log')
    parser.add_argument('--executor', help='Native VE executable; defaults to built optimized projection, otherwise session baseline')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE',
               VE_LD_LIBRARY_PATH='/opt/nec/ve/ncc/5.4.1/lib:/opt/nec/ve/nfort/5.4.1/lib:/opt/nec/ve/nlc/3.1.0/lib')
    default_executor = root/'build/qwen15-projection/qwen-infer-ve'
    if not default_executor.is_file():
        default_executor = root/'build/qwen15-session/qwen-infer-ve'
    executor = args.executor or str(default_executor)
    command = ['ve_exec','-N',str(args.slot),executor,
               '--model',args.model,'--threads',str(args.threads),'--context','2048',
               '--no-mmap','--dense-cache-mib','16384','--serve']
    with open(args.stderr_log,'w') as err:
        session = QwenSession(command,err,env)
        try:
            print(json.dumps(dict(session.ready, host_startup_seconds=session.startup_seconds)), flush=True)
            for prompt in args.prompt:
                print(json.dumps(session.request(prompt,args.tokens), ensure_ascii=False), flush=True)
        finally:
            if session.close():
                raise RuntimeError('native executor exited unsuccessfully; inspect local stderr log')


if __name__ == '__main__':
    main()
