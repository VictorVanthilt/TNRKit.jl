using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

bound_trg_blk(steps::Int, data) = abs(log(data[end][1]) * 4.0^((1-steps)))
bound_trg_env(steps::Int, data) = abs(log(data[end][2]) * 2.0^((1-steps)))
stopping_criterion = convcrit(1e-20, bound_trg_blk)&maxiter(40)&convcrit(1e-20, bound_trg_env)
#stopping_criterion = convcrit(1e-20, bound_trg_blk)&maxiter(40)
Ising_βc = log(1.0+sqrt(2))/2.0

T = classical_ising(1)
E1 = TensorMap(ones, ℂ^1⊗ℂ^2←ℂ^1)
E2 = TensorMap(ones, ℂ^1←ℂ^1⊗ℂ^2)
scheme = Boundary_TRG(T, E1, E2)
data_boundtrg = run!(scheme, truncdim(16), stopping_criterion; finalize_beginning=true)