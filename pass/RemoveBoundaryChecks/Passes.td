#ifndef TRITON_EXT_PASS_REMOVEBOUNDARYCHECKS
#define TRITON_EXT_PASS_REMOVEBOUNDARYCHECKS

include "mlir/Pass/PassBase.td"

def TritonIntelExtRemoveBoundaryChecks
    : Pass<"triton-intel-ext-remove-boundary-checks", "mlir::ModuleOp"> {
  let summary = "Remove unnecessary boundary checks from load operations (block pointers only)";

  let description = [{
    This pass attempts to remove boundary checks that aren't necessary in a tt.load operation.
    For example, given:
    %lb = arith.bitcast %c0_i32 : i32 to i32
    %ub = arith.bitcast %c1024_i32 : i32 to i32
    %st = arith.bitcast %c64_i32 : i32 to i32
    scf.for %iv = %lb to %ub step %st : i32 {
      %s0 = arith.constant 512 : i64
      %s1 = arith.constant 64 : i64
      %s2 = arith.constant 1024 : i64
      %a = arith.constant 65536 : i64
      %b = arith.constant 1 : i64
      %b = arith.constant 64 : i64
      %y = arith.constant 0 : i32
      %ptr = tt.make_tensor_ptr %base, [%s0, %s1, %s2], [%a, %b, %c], [%x, %y, %iv]
                                {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
      %load = tt.load %ptr {boundaryCheck = array<i32: 2>} : !tt.ptr<tensor<1x512x64xf16>>
      ...
      // here %ptr is never updated.
    }

    The transformation would drop the boundary check on the load operation because:
      - `%ptr` is never advanced in the loop
      - `%iv` has values [0, 64, 128, ..., 960], max(%iv) = 960
      - `%s2` is equal to 1024
      - the boundary check expression `max(%iv) + load_res.shape_in_dim -1` < `%s2` is true.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];

  let constructor = "mlir::triton_plugin::passes::createRemoveBoundaryChecksPass()";
}

#endif
