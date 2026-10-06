"""One bounded local native worker holding all block weights across requests.

The worker inherits its caller's process group for temperature supervision.
No sockets, global settings, external request paths or generation cache.
"""
from pathlib import Path
import os
import select
import struct
import subprocess
import tempfile
import time

import numpy as np
import torch
from gpt_neo_ve import GPTNeoNative, export_block
from ve_transformer import ROOT


@torch.inference_mode()
def export_bundle(model, path, capacity):
    blocks = model.transformer.h
    width = model.config.hidden_size
    if (type(capacity) is not int or not 1 <= capacity <= min(1024, model.config.max_position_embeddings)
            or not 1 <= len(blocks) <= 32):
        raise ValueError('unsupported model capacity or layer count')
    first = blocks[0]
    for block in blocks:
        if (block.attn.attention.num_heads != first.attn.attention.num_heads
                or block.mlp.c_fc.out_features != first.mlp.c_fc.out_features):
            raise ValueError('resident worker requires uniform head and FFN dimensions')
    f = first.mlp.c_fc.out_features
    parameters = len(blocks)*(4*width*width+2*width*f+9*width+f+width)
    if parameters > 64*1024*1024:
        raise ValueError('resident model exceeds bounded parameter storage')
    dummy = torch.zeros(1, width)
    with tempfile.TemporaryDirectory(prefix='bundle-export-', dir=ROOT/'build') as tmp:
        with open(path, 'wb') as output:
            output.write(b'VTGPTN1\0'+struct.pack('<3I', len(blocks), width, capacity))
            for index, block in enumerate(blocks):
                source = Path(tmp)/('block-{}.vtf'.format(index))
                export_block(block, model.config, dummy, source)
                output.write(source.read_bytes())


class ResidentSession:
    def __init__(self, model, *, node=1, backend='fast', capacity=512, timeout=15, threads=1):
        if backend not in ('cpu', 'fast') or type(node) is not int or node < 0:
            raise ValueError('resident backend must be cpu or fast, node nonnegative')
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError('threads must be 1..8')
        if not 0.05 <= timeout <= 60:
            raise ValueError('timeout must be 0.05..60 seconds')
        self.timeout, self.capacity = timeout, capacity
        self.width = model.config.hidden_size
        self.proc = None
        self._tmp = self._stderr = None
        self.closed = False
        self.trace = []
        binary = ROOT/'build'/('resident-cpu' if backend == 'cpu' else 'resident-ve-fast')
        if not binary.is_file():
            raise FileNotFoundError('build workers using make resident')
        try:
            self._tmp = tempfile.TemporaryDirectory(prefix='resident-', dir=ROOT/'build')
            bundle = Path(self._tmp.name)/'model.bundle'
            start = time.perf_counter()
            export_bundle(model, bundle, capacity)
            self.export_ms = (time.perf_counter()-start)*1000
            env = os.environ.copy()
            env['VE_LD_LIBRARY_PATH'] = ':'.join([
                '/opt/nec/ve/ncc/5.4.1/lib', '/opt/nec/ve/nfort/5.4.1/lib',
                '/opt/nec/ve/nlc/3.1.0/lib'] +
                ([env['VE_LD_LIBRARY_PATH']] if env.get('VE_LD_LIBRARY_PATH') else []))
            env['OMP_NUM_THREADS'] = str(threads)
            env['OMP_DYNAMIC'] = 'FALSE'
            command = [str(binary), str(bundle)]
            if backend == 'fast':
                command = ['/opt/nec/ve/bin/ve_exec', '-N', str(node)]+command
            self._stderr = tempfile.TemporaryFile(dir=ROOT/'build')
            start = time.perf_counter()
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self._stderr, env=env, cwd=ROOT, bufsize=0)
            os.set_blocking(self.proc.stdin.fileno(), False)
            os.set_blocking(self.proc.stdout.fileno(), False)
            ready = self._read_exact(16, time.monotonic()+timeout)
            magic, self.load_ms = struct.unpack('<8sd', ready)
            if magic != b'VTREADY1' or not np.isfinite(self.load_ms) or self.load_ms < 0:
                raise RuntimeError('invalid worker handshake')
            self.start_ms = (time.perf_counter()-start)*1000
        except BaseException:
            self.close()
            raise

    def _read_exact(self, count, deadline):
        result = bytearray()
        while len(result) < count:
            left = deadline-time.monotonic()
            if left <= 0 or not select.select([self.proc.stdout], [], [], max(0,left))[0]:
                raise TimeoutError('resident response timed out')
            try:
                chunk = os.read(self.proc.stdout.fileno(), count-len(result))
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError('resident worker closed its output')
            result.extend(chunk)
        return bytes(result)

    def _write_all(self, data, deadline):
        view = memoryview(data)
        while view:
            left = deadline-time.monotonic()
            if left <= 0 or not select.select([], [self.proc.stdin], [], max(0,left))[1]:
                raise TimeoutError('resident request timed out')
            try:
                count = os.write(self.proc.stdin.fileno(), view)
            except BlockingIOError:
                continue
            if not count:
                raise RuntimeError('resident worker closed its input')
            view = view[count:]

    @torch.inference_mode()
    def forward(self, x, *, last_only=False):
        if self.closed or self.proc is None or self.proc.poll() is not None:
            self.close()
            raise RuntimeError('resident session is closed')
        if (x.device.type != 'cpu' or x.dtype != torch.float32 or x.ndim != 2
                or x.requires_grad or not 1 <= x.shape[0] <= self.capacity
                or x.shape[1] != self.width or not torch.isfinite(x).all()):
            raise ValueError('finite bounded CPU float32 features required')
        output_tokens = 1 if last_only else x.shape[0]
        data = struct.pack('<2I', x.shape[0], int(bool(last_only))) + np.asarray(
            x.contiguous().numpy(), dtype='<f4').tobytes()
        start = time.perf_counter()
        deadline = time.monotonic()+self.timeout
        try:
            self._write_all(data, deadline)
            header = self._read_exact(24, deadline)
            magic, status, tokens, compute_ms = struct.unpack('<8sIId', header)
            if (magic != b'VTRES01\0' or status or tokens != output_tokens
                    or not np.isfinite(compute_ms) or compute_ms < 0):
                raise RuntimeError('invalid resident response')
            result = self._read_exact(tokens*self.width*4, deadline)
            output = np.frombuffer(result, dtype='<f4').reshape(tokens, self.width).copy()
            if not np.isfinite(output).all():
                raise RuntimeError('nonfinite resident output')
            self.trace.append(dict(tokens=x.shape[0], compute_ms=compute_ms,
                                   roundtrip_ms=(time.perf_counter()-start)*1000,
                                   sent_bytes=len(data), received_bytes=24+len(result)))
            if len(self.trace) > 256:
                del self.trace[0]
            return torch.from_numpy(output)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self._write_all(struct.pack('<2I', 0, 0), time.monotonic()+0.2)
                    self.proc.wait(timeout=2)
            except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait()
            finally:
                self.proc.stdin.close()
                self.proc.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class GPTNeoResident(GPTNeoNative):
    def __init__(self, model, node=1, backend='fast', capacity=512, threads=1):
        super().__init__(model, node, backend)
        self.session = ResidentSession(model, node=node, backend=backend, capacity=capacity, threads=threads)

    @torch.inference_mode()
    def logits(self, ids, *, last_only=False, check_layers=None):
        if check_layers is not None:
            raise ValueError('resident response contains final features, not per-layer features')
        x = self.features(ids)
        x = self.session.forward(x, last_only=last_only)
        return self.model.lm_head(self.model.transformer.ln_f(x))

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
