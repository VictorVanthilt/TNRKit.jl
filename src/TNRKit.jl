module TNRKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf
using KrylovKit
using OptimKit, Zygote
using PEPSKit: network_value, InfinitePartitionFunction, CTMRGEnv
using DocStringExtensions

# stop criteria
include("utility/stopping.jl")
export maxiter, convcrit
export trivial_convcrit

# schemes
include("schemes/tnrscheme.jl")
include("schemes/trg.jl")
include("schemes/btrg.jl")
include("schemes/hotrg.jl")
include("schemes/hotrg3d.jl")
include("schemes/atrg.jl")
include("schemes/atrg3d.jl")
include("schemes/impurityhotrg.jl")
# CTM methods
include("schemes/ctm/utility.jl")
include("schemes/ctm/c4ctm.jl")
include("schemes/ctm/rctm.jl")
include("schemes/ctm/ctm_trg.jl")
include("schemes/ctm/ctm_hotrg.jl")
include("schemes/ctm/onesite_ctm.jl")
include("schemes/ctm/sublattice_ctm.jl")

# Loop Methods
include("schemes/looptnr.jl")
include("schemes/symmetric_looptnr.jl")
export classical_ising_inv # Ising model with all legs in codomain

export TNRScheme

export TRG
export BTRG
export HOTRG
export HOTRG_3D
export ATRG
export ATRG_3D
export ImpurityHOTRG

export CTM
export Sublattice_CTM
export c4CTM
export rCTM
export ctm_TRG
export ctm_HOTRG
export lnz

export LoopTNR
export SLoopTNR

export run!

# models
include("models/ising.jl")
export classical_ising, classical_ising_symmetric, ising_βc, f_onsager, ising_cft_exact,
    ising_βc_3D, classical_ising_symmetric_3D, classical_ising_3D

include("models/gross-neveu.jl")
export gross_neveu_start

include("models/sixvertex.jl")
export sixvertex

include("models/potts.jl")
export classical_potts, classical_potts_symmetric, potts_βc

include("models/clock.jl")
export classical_clock

# utility functions
include("utility/free_energy.jl")
export free_energy

include("utility/cft.jl")
export cft_data, central_charge, cft_data!

include("utility/finalize.jl")
export finalize!, finalize_two_by_two!, finalize_cftdata!, finalize_central_charge!

include("utility/cdl.jl")
export cdl_tensor

include("utility/projectors.jl")

include("utility/blocking.jl")
export block_tensors
end
