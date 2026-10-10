"""Read-only validated-prefix adapter for the locally tested VEOS ABI.

The oversized aligned output is not a claim of a complete structure layout.
Only the total/used/free prefix is consumed and numerically validated.
"""
import ctypes
import os
import subprocess

LIBRARY='/opt/nec/ve/veos/lib64/libveosinfo.so.3'
class VeMemoryInfo:
    def __init__(self):
        if os.environ.get('VE_TRANSFORMER_TEMPERATURE_SUPERVISED')!='1':
            raise RuntimeError('temperature supervision required for hardware query')
        version=subprocess.check_output(['rpm','-q','veosinfo3','--qf','%{VERSION}'],text=True)
        if version!='3.6.0' or ctypes.sizeof(ctypes.c_ulong)!=8:
            raise RuntimeError('unvalidated VEOS library version or unsigned-long width')
        self.version=version
        self.library=ctypes.CDLL(LIBRARY,use_errno=True)
        self.read=self.library.ve_mem_info
        self.read.argtypes=[ctypes.c_int,ctypes.c_void_p]
        self.read.restype=ctypes.c_int
    def sample(self,node):
        if node not in (1,2,3):raise ValueError('known runtime node required')
        output=(ctypes.c_ulong*1024)()
        ctypes.set_errno(0)
        if self.read(node,ctypes.byref(output)):
            raise RuntimeError('VEOS memory query failed')
        total,used,free=map(int,output[:3])
        if total!=48*1024**2 or total!=used+free or not 0<=used<=total:
            raise RuntimeError('VEOS memory prefix invariant failed')
        return {'total_kib':total,'used_kib':used,'free_kib':free}
