module TNRKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf
using Zygote, OptimKit, KrylovKit

# stop criteria
include("utility/stopping.jl")
export maxiter, convcrit
export trivial_convcrit

# schemes
include("schemes/tnrscheme.jl")
include("schemes/trg.jl")
include("schemes/btrg.jl")
include("schemes/hotrg.jl")
include("schemes/gilt.jl")
include("schemes/gilt-tnr.jl")
include("schemes/atrg.jl")
include("schemes/slooptnr.jl")
include("schemes/looptnr.jl")

export TNRScheme
export TRG, trg_convcrit
export BTRG, btrg_convcrit
export HOTRG, hotrg_convcrit
export ATRG, atrg_convcrit
export GILTTNR, gilttnr_convcrit
export SLoopTNR
export LoopTNR, entanglement_filtering!, loop_opt!, step!

export run!

# models
include("models/ising.jl")
export classical_ising, classical_ising_symmetric, Potts_βc, Ising_βc, f_onsager, Free_energy_Ising

include("models/gross-neveu.jl")
export gross_neveu_start

include("models/sixvertex.jl")
export sixvertex

# utility functions
include("utility/cft.jl")
export cft_data, central_charge

include("utility/finalize.jl")
export finalize!, finalize_two_by_two!
end
