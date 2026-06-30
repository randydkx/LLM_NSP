
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_ma-user'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.enable_cpp_symbolic_shape_guards = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.8.0+cpu
# torch cuda version: None
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        npu_dtype_cast = torch.ops.npu.npu_dtype_cast.default(arg2_1, torch.float32)
        amax = torch.ops.aten.amax.default(npu_dtype_cast, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(npu_dtype_cast, amax);  npu_dtype_cast = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        _npu_dtype_cast = torch.ops.npu._npu_dtype_cast.default(arg2_1, torch.float32)
        amax_1 = torch.ops.aten.amax.default(_npu_dtype_cast, [-1], True)
        abs_1 = torch.ops.aten.abs.default(amax_1)
        eq_11 = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(eq_11, full_default, amax_1);  eq_11 = full_default = amax_1 = None
        squeeze = torch.ops.aten.squeeze.dims(where, [-1])
        sub_7 = torch.ops.aten.sub.Tensor(_npu_dtype_cast, where);  _npu_dtype_cast = where = None
        exp_1 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1]);  exp_1 = None
        log = torch.ops.aten.log.default(sum_2);  sum_2 = None
        add_9 = torch.ops.aten.add.Tensor(log, squeeze);  log = squeeze = None
        mul_8 = torch.ops.aten.mul.Tensor(div, arg2_1);  div = arg2_1 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_8, [-1], dtype = torch.float32);  mul_8 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_9, sum_3);  add_9 = sum_3 = None
        return (sub_12,)
        
def load_args(reader):
    reader.symint(15206)  # arg0_1
    reader.symint(152064)  # arg1_1
    buf0 = reader.storage(None, 2*s57*(s1 - 1) + 2*s57, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (s1, s57), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)