module TRGKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf

# stop criteria
include("utility/stopping.jl")
export maxiter, convcrit

# schemes
include("schemes/trgscheme.jl")
include("schemes/trg.jl")
include("schemes/btrg.jl")
include("schemes/hotrg.jl")
include("schemes/gilt.jl")
include("schemes/gilt-tnr.jl")
include("schemes/atrg.jl")

export TRGScheme
export TRG, trg_convcrit
export BTRG, btrg_convcrit
export HOTRG, hotrg_convcrit
export ATRG, atrg_convcrit
export GILTTNR, gilttnr_convcrit

export run!

# models
include("models/ising.jl")
export classical_ising, classical_ising_symmetric, Potts_βc, Ising_βc

include("models/gross-neveu.jl")
export gross_neveu_start

# utility functions
include("utility/cft.jl")
export cft_data, central_charge
end
