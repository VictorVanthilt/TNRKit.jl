module TNRKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf
using KrylovKit
using PEPSKit: network_value, InfinitePartitionFunction, CTMRGEnv

# stop criteria
include("utility/stopping.jl")
export maxiter, convcrit
export trivial_convcrit

# schemes
include("schemes/tnrscheme.jl")
include("utility/tnrutilities.jl")
include("schemes/trg.jl")
include("schemes/btrg.jl")
include("schemes/hotrg.jl")
include("schemes/atrg.jl")
include("schemes/looptnr.jl")
include("schemes/c4ctm.jl")
include("schemes/rctm.jl")
include("schemes/ctmhotrg.jl")
include("schemes/efbtrg.jl")

export TNRScheme
export QR_L, QR_R, find_L, find_R, P_decomp, SVD12, find_projectors, cost_func
export TRG
export BTRG
export HOTRG
export ATRG
export LoopTNR, loop_opt!
export c4CTM
export rCTM
export CTMHOTRG
export EFBTRG, entanglement_filtering!

export run!

# models
include("models/ising.jl")
export classical_ising, classical_ising_symmetric, potts_βc, ising_βc, f_onsager

include("models/gross-neveu.jl")
export gross_neveu_start

include("models/sixvertex.jl")
export sixvertex

# utility functions
include("utility/cft.jl")
export cft_data, central_charge

include("utility/finalize.jl")
export finalize!, finalize_two_by_two!, finalize_cftdata!, finalize_central_charge!

include("utility/cdl.jl")
export cdl_tensor
end
