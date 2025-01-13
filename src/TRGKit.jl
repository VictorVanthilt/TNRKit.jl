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

export TRGScheme, TRG, BTRG, HOTRG, GILT, trg_convcrit, btrg_convcrit, hotrg_convcrit
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
