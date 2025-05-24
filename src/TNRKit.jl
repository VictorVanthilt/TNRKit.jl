module TNRKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf
using KrylovKit
using OptimKit, Zygote
using PEPSKit: network_value, InfinitePartitionFunction, CTMRGEnv

# stop criteria
include("utility/stopping.jl")
export maxiter, convcrit
export trivial_convcrit

# schemes
include("schemes/tnrscheme.jl")
include("schemes/trg.jl")
include("schemes/btrg.jl")
include("schemes/hotrg.jl")
include("schemes/atrg.jl")
include("schemes/atrg3d.jl")

# CTM methods
include("schemes/ctm/utility.jl")
include("schemes/ctm/c4ctm.jl")
export c4CTM
include("schemes/ctm/rctm.jl")
export rCTM
# include("schemes/ctm/ctmhotrg.jl"); export CTMHOTRG
include("schemes/ctm/ctm_hotrg.jl")
export ctm_HOTRG
include("schemes/ctm/ctm_trg.jl")
export ctm_TRG

# Loop Methods
include("schemes/looptnr.jl")
include("schemes/symmetric_looptnr.jl")
export classical_ising_inv # Ising model with all legs in codomain

export TNRScheme

export TRG
export BTRG
export HOTRG
export ATRG
export ATRG_3D

export c4CTM
export rCTM
export CTMHOTRG

export LoopTNR
export SLoopTNR

export run!

# models
include("models/ising.jl")
export classical_ising, classical_ising_symmetric, ising_βc, f_onsager,
       ising_βc_3D, classical_ising_symmetric_3D, classical_ising_3D

include("models/gross-neveu.jl")
export gross_neveu_start

include("models/sixvertex.jl")
export sixvertex

include("models/potts.jl")
export classical_potts, classical_potts_symmetric, potts_βc

# utility functions
include("utility/cft.jl")
export cft_data, central_charge

include("utility/finalize.jl")
export finalize!, finalize_two_by_two!, finalize_cftdata!, finalize_central_charge!

include("utility/cdl.jl")
export cdl_tensor
end
