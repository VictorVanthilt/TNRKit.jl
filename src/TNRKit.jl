module TNRKit
using TensorKit, LinearAlgebra
using LoggingExtras, Printf
using KrylovKit
using OptimKit, Zygote
using DocStringExtensions
using SpecialFunctions
using FastGaussQuadrature
using QuadGK
using Base.Threads
using Combinatorics: permutations

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
# CTM methods
include("schemes/ctm/utility.jl")
include("schemes/ctm/c4vctm.jl")
include("schemes/ctm/rctm.jl")
include("schemes/ctm/ctm_trg.jl")
include("schemes/ctm/ctm_hotrg.jl")
include("schemes/ctm/onesite_ctm.jl")
include("schemes/ctm/sublattice_ctm.jl")
include("schemes/ctm/triangular.jl")
include("schemes/ctm/ctm_triangular.jl")
include("schemes/ctm/c6vctm_triangular.jl")

# Impurity methods
include("schemes/impuritytrg.jl")
include("schemes/impurityhotrg.jl")

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

export CTM
export Sublattice_CTM
export c4vCTM
export rCTM
export ctm_TRG
export ctm_HOTRG
export lnz
export c6vCTM_triangular
export CTM_triangular

export ImpurityTRG
export ImpurityHOTRG

export LoopTNR
export SLoopTNR

export run!

# models
include("models/ising.jl")
include("models/ising_triangular.jl")
export classical_ising, classical_ising_symmetric, ising_βc, f_onsager, ising_cft_exact,
    ising_βc_3D, classical_ising_symmetric_3D, classical_ising_3D, classical_ising_impurity,
    classical_ising_triangular, classical_ising_triangular_symmetric,
    ising_βc_triangular, f_onsager_triangular

include("models/gross-neveu.jl")
export gross_neveu_start

include("models/sixvertex.jl")
export sixvertex

include("models/potts.jl")
export classical_potts, classical_potts_symmetric, potts_βc, classical_potts_impurity

include("models/clock.jl")
export classical_clock

include("models/XY.jl")
export classical_XY_U1_symmetric
export classical_XY_O2_symmetric

include("models/phi4_real.jl")
export phi4_real, phi4_real_imp1, phi4_real_imp2
export phi4_real_symmetric

include("models/phi4_complex.jl")
export phi4_complex, phi4_complex_impϕ, phi4_complex_impϕdag, phi4_complex_impϕabs, phi4_complex_impϕ2, phi4_complex_all
export phi4_complex_symmetric

# utility functions
include("utility/free_energy.jl")
export free_energy

include("utility/cft.jl")
export cft_data, central_charge, ground_state_degeneracy, gu_wen_ratio

include("utility/finalize.jl")
export Finalizer, two_by_two_Finalizer, finalize!, finalize_two_by_two!, finalize_cftdata!, finalize_central_charge!,
    finalize_groundstatedegeneracy!, GSDegeneracy_Finalizer, guwenratio_Finalizer

include("utility/cdl.jl")
export cdl_tensor

include("utility/projectors.jl")

include("utility/blocking.jl")
export block_tensors

include("utility/network_value.jl")
end
