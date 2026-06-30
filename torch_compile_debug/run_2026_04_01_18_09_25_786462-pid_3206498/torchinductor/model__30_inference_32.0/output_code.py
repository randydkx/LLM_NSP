# AOT ID: ['30_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import torch_npu
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
import torch_npu
has_initialized = False
from torch_npu._inductor import get_current_raw_stream as get_raw_stream
from torch_npu._inductor import get_current_raw_stream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_ma-user/yg/cyghmxihyztebcezwvtgxgklbqeyaui5o5nvhtzynmqggt7p5qgj.py
# Topologically Sorted Source Nodes: [logsumexp, pd, mul, sum_1, entropy], Original ATen: [npu._npu_dtype_cast, aten.logsumexp, npu.npu_dtype_cast, aten._softmax, aten.mul, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   entropy => sub_12
#   logsumexp => _npu_dtype_cast, abs_1, add_9, amax_1, eq_11, exp_1, full_default, log, sub_7, sum_2, where
#   mul => mul_8
#   pd => amax, div, exp, npu_dtype_cast, sub_2, sum_1
#   sum_1 => sum_3
# Graph fragment:
#   %_npu_dtype_cast : [num_users=2] = call_function[target=torch.ops.npu._npu_dtype_cast.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %amax_1 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%_npu_dtype_cast, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_1,), kwargs = {})
#   %eq_11 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: npu:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_11, %full_default, %amax_1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_npu_dtype_cast, %where), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_7,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_2,), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %npu_dtype_cast : [num_users=2] = call_function[target=torch.ops.npu.npu_dtype_cast.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%npu_dtype_cast, [-1], True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%npu_dtype_cast, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %arg2_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [-1]), kwargs = {dtype: torch.float32})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %sum_3), kwargs = {})
# SchedulerNodes: [SchedulerNode(name='op0'), SchedulerNode(name='op1'), SchedulerNode(name='op2'), SchedulerNode(name='op3'), SchedulerNode(name='op4'), SchedulerNode(name='op5')]

triton_unk_fused__npu_dtype_cast__softmax_logsumexp_mul_npu_dtype_cast_sub_sum_0 = async_compile.triton('triton_unk_fused__npu_dtype_cast__softmax_logsumexp_mul_npu_dtype_cast_sub_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._inductor.runtime import triton_helpers
from torch_npu._inductor import npu_triton_heuristics
from torch_npu._inductor import npu_triton_helpers
from torch_npu._inductor.runtime import NPUDeviceProperties
from torch_npu._inductor.npu_triton_helpers import libdevice, math as tl_math
import torch
import torch_npu

@npu_triton_heuristics.reduction_npu_index(
    size_hints=[15245, 152064],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'ks0': 'i64', 'x0_numel': 'i64', 'r1_numel': 'i64'}, 'device': NPUDeviceProperties(type='npu', index=0, multi_processor_count=48, cc='Ascend910B2', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_unk_fused__npu_dtype_cast__softmax_logsumexp_mul_npu_dtype_cast_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1'], 'backend_hash': '979696580158830a583bc7fba81ff8ee2cc5867124830081a9241511c8991f6f', 'split_axis': [0], 'tiling_axis': [0, 1], 'axis_names': ['x0', 'r1'], 'low_dims': {1}, 'numof_reduction_axis': 1, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'store_cubin': False, 'force_disable_caches': False, 'profile_bandwidth_with_do_bench_using_profiling': False}
)
@triton.jit
def triton_unk_fused__npu_dtype_cast__softmax_logsumexp_mul_npu_dtype_cast_sub_sum_0(in_out_ptr1, in_ptr0, ks0, x0_numel, r1_numel, X0BLOCK : tl.constexpr, X0BLOCK_SUB : tl.constexpr, R1BLOCK_SUB : tl.constexpr):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0= tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1= tl.arange(0, R1BLOCK_SUB)
    loops_r1 = (r1_numel + R1BLOCK_SUB - 1) // R1BLOCK_SUB
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0[:,None]
        x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
        _tmp3 = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], float("-inf"), tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp0 = tl.load(in_ptr0 + (r1 + ks0*x0), r1_mask & x0_mask, other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tl.reshape(tmp1, [X0BLOCK_SUB, R1BLOCK_SUB])
            tmp4 = triton_helpers.maximum(_tmp3, tmp2)
            _tmp3 = tl.where(r1_mask & x0_mask, tmp4, _tmp3)
        tmp3 = tl.max(_tmp3, 1).reshape(X0BLOCK_SUB, 1)
        _tmp15 = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], 0, tl.float32)
        _tmp18 = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], float("-inf"), tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp5 = tl.load(in_ptr0 + (r1 + ks0*x0), r1_mask & x0_mask, other=0.0).to(tl.float32)
            tmp6 = tmp5.to(tl.float32)
            tmp7 = tl_math.abs(tmp3)
            tmp8 = float("inf")
            tmp9 = tmp7 == tmp8
            tmp10 = 0.0
            tmp11 = tl.where(tmp9, tmp10, tmp3)
            tmp12 = tmp6 - tmp11
            tmp13 = tl_math.exp(tmp12)
            tmp14 = tl.reshape(tmp13, [X0BLOCK_SUB, R1BLOCK_SUB])
            tmp16 = _tmp15 + tmp14
            _tmp15 = tl.where(r1_mask & x0_mask, tmp16, _tmp15)
            tmp17 = tl.reshape(tmp6, [X0BLOCK_SUB, R1BLOCK_SUB])
            tmp19 = triton_helpers.maximum(_tmp18, tmp17)
            _tmp18 = tl.where(r1_mask & x0_mask, tmp19, _tmp18)
        tmp15 = tl.sum(_tmp15, 1).reshape(X0BLOCK_SUB, 1)
        tmp18 = tl.max(_tmp18, 1).reshape(X0BLOCK_SUB, 1)
        _tmp25 = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp20 = tl.load(in_ptr0 + (r1 + ks0*x0), r1_mask & x0_mask, other=0.0).to(tl.float32)
            tmp21 = tmp20.to(tl.float32)
            tmp22 = tmp21 - tmp18
            tmp23 = tl_math.exp(tmp22)
            tmp24 = tl.reshape(tmp23, [X0BLOCK_SUB, R1BLOCK_SUB])
            tmp26 = _tmp25 + tmp24
            _tmp25 = tl.where(r1_mask & x0_mask, tmp26, _tmp25)
        tmp25 = tl.sum(_tmp25, 1).reshape(X0BLOCK_SUB, 1)
        _tmp34 = tl.full([X0BLOCK_SUB, R1BLOCK_SUB], 0, tl.float32)
        for loop_r1 in range(loops_r1):
            r1 = (loop_r1 * R1BLOCK_SUB) + base_r1[None,:]
            r1_mask = r1 < r1_numel
            tmp27 = tl.load(in_ptr0 + (r1 + ks0*x0), r1_mask & x0_mask, other=0.0).to(tl.float32)
            tmp28 = tmp27.to(tl.float32)
            tmp29 = tmp28 - tmp18
            tmp30 = tl_math.exp(tmp29)
            tmp31 = (tmp30 / tmp25)
            tmp32 = tmp31 * tmp28
            tmp33 = tl.reshape(tmp32, [X0BLOCK_SUB, R1BLOCK_SUB])
            tmp35 = _tmp34 + tmp33
            _tmp34 = tl.where(r1_mask & x0_mask, tmp35, _tmp34)
        tmp34 = tl.sum(_tmp34, 1).reshape(X0BLOCK_SUB, 1)
        tmp36 = tl_math.log(tmp15)
        tmp37 = tl_math.abs(tmp3)
        tmp38 = float("inf")
        tmp39 = tmp37 == tmp38
        tmp40 = 0.0
        tmp41 = tl.where(tmp39, tmp40, tmp3)
        tmp42 = tmp36 + tmp41
        tmp43 = tmp42 - tmp34
        tl.store(in_out_ptr1 + (x0), tmp43, x0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s1 = arg0_1
    s57 = arg1_1
    buf1 = empty_strided((s1, ), (1, ), device='npu', dtype=torch.float32)
    buf5 = buf1;   # reuse
    # Topologically Sorted Source Nodes: [logsumexp, pd, mul, sum_1, entropy], Original ATen: [npu._npu_dtype_cast, aten.logsumexp, npu.npu_dtype_cast, aten._softmax, aten.mul, aten.sum, aten.sub]
    stream0 = get_raw_stream(0)
    triton_unk_fused__npu_dtype_cast__softmax_logsumexp_mul_npu_dtype_cast_sub_sum_0.run(buf5, arg2_1, s57, 15245, 152064, stream=stream0)

    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 15245
    arg1_1 = 152064
    arg2_1 = rand_strided((15245, 152064), (152064, 1), device='npu:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
