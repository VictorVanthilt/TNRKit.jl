module TNRKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf

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

export TNRScheme
export TRG
export BTRG
export HOTRG
export ATRG

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
export finalize!, finalize_two_by_two!
end
