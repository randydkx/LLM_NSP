class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s1)", arg1_1: "Sym(s57)", arg2_1: "bf16[s1, s57]"):
         # File: /home/ma-user/work/hw_nsp/NullspaceOfLLM/verl/utils/torch_functional.py:236 in entropy_from_logits, code: pd = torch.nn.functional.softmax(logits, dim=-1)
        npu_dtype_cast: "f32[s1, s57]" = torch.ops.npu.npu_dtype_cast.default(arg2_1, torch.float32)
        amax: "f32[s1, 1]" = torch.ops.aten.amax.default(npu_dtype_cast, [-1], True)
        sub_2: "f32[s1, s57]" = torch.ops.aten.sub.Tensor(npu_dtype_cast, amax);  npu_dtype_cast = amax = None
        exp: "f32[s1, s57]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[s1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[s1, s57]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
         # File: /home/ma-user/work/hw_nsp/NullspaceOfLLM/verl/utils/torch_functional.py:237 in entropy_from_logits, code: entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
        _npu_dtype_cast: "f32[s1, s57]" = torch.ops.npu._npu_dtype_cast.default(arg2_1, torch.float32)
        amax_1: "f32[s1, 1]" = torch.ops.aten.amax.default(_npu_dtype_cast, [-1], True)
        abs_1: "f32[s1, 1]" = torch.ops.aten.abs.default(amax_1)
        eq_11: "b8[s1, 1]" = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[s1, 1]" = torch.ops.aten.where.self(eq_11, full_default, amax_1);  eq_11 = full_default = amax_1 = None
        squeeze: "f32[s1]" = torch.ops.aten.squeeze.dims(where, [-1])
        sub_7: "f32[s1, s57]" = torch.ops.aten.sub.Tensor(_npu_dtype_cast, where);  _npu_dtype_cast = where = None
        exp_1: "f32[s1, s57]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_2: "f32[s1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1]);  exp_1 = None
        log: "f32[s1]" = torch.ops.aten.log.default(sum_2);  sum_2 = None
        add_9: "f32[s1]" = torch.ops.aten.add.Tensor(log, squeeze);  log = squeeze = None
        mul_8: "f32[s1, s57]" = torch.ops.aten.mul.Tensor(div, arg2_1);  div = arg2_1 = None
        sum_3: "f32[s1]" = torch.ops.aten.sum.dim_IntList(mul_8, [-1], dtype = torch.float32);  mul_8 = None
        sub_12: "f32[s1]" = torch.ops.aten.sub.Tensor(add_9, sum_3);  add_9 = sum_3 = None
        return (sub_12,)
        