using Revise, TensorKit
includet("../src/TRGKit.jl")
using .TRGKit

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1e-16, trg_f) & maxiter(20)

# choose a TensorKit truncation scheme
trunc = truncdim(16) & truncbelow(1e-40)

# initialize the TRG scheme
scheme = TRG(classical_ising(1.0))

# run the TRG scheme (and normalize and store the norm in the beginning (finalize_beginning=true))
data = run!(scheme, trunc, stopping_criterion; finalize_beginning=true)
# or: data = run!(scheme, truncdim(16)), this will default to maxiter(100)

# initialize the BTRG scheme
scheme = BTRG(classical_ising(1.0), -0.5)

# run the BTRG scheme
data = run!(scheme, trunc, stopping_criterion)

# initialize the HOTRG scheme
scheme = HOTRG(classical_ising(1.0))

# run the HOTRG scheme
data = run!(scheme, trunc, stopping_criterion)
