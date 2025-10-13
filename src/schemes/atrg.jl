"""
$(TYPEDEF)

Anisotropic Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T [, finalize=finalize!])

### Running the algorithm
    run!(::ATRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields
 
$(TYPEDFIELDS)

### References
* [Adachi et. al. Phys. Rev. B 102 (2020)](@cite adachiAnisotropicTensorRenormalization2020)
"""
mutable struct ATRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function ATRG(T::TensorMap{E, S, 2, 2}; finalize = (finalize!)) where {E, S}
        return new(T, finalize)
    end
end

function step!(scheme::ATRG, trunc::TensorKit.TruncationScheme)
    _step!(scheme, trunc)

    scheme.T = permute(scheme.T, ((2, 4), (1, 3)))

    _step!(scheme, trunc)

    return scheme.T = permute(scheme.T, ((3, 1), (4, 2)))
end

function _step!(scheme::ATRG, trunc::TensorKit.TruncationScheme)
    A, S, B, _ = tsvd(scheme.T, ((1, 3), (2, 4)); trunc = trunc)
    C, D = deepcopy.([A, B])

    @tensor begin
        B[-1; -2 -3] = S[-1; 1] * B[1; -2 -3]
        C[-1 -2; -3] = C[-1 -2; 1] * S[1; -3]
    end

    @tensor M[-1 -2; -3 -4] := B[-3; 1 -4] * C[-1 1; -2]

    X, S, Y, _ = tsvd(M, ((1, 3), (2, 4)); trunc = trunc)
    sqrtS = sqrt(S)

    @tensor begin
        X[-1 -2; -3] = X[-1 -2; 1] * sqrtS[1; -3]
        Y[-1; -2 -3] = sqrtS[-1; 1] * Y[1; -2 -3]
    end

    @tensor Q[-1 -2; -3 -4] := A[3 -3; 2] * D[1; -2 4] * X[4 2; -4] * Y[-1 1; 3]

    H, S, G, _ = tsvd(Q; trunc = trunc)
    sqrtS = sqrt(S)

    @tensor begin
        H[-1 -2; -3] = H[-1 -2; 1] * sqrtS[1; -3]
        G[-1; -2 -3] = sqrtS[-1; 1] * G[1; -2 -3]
    end

    return @tensor scheme.T[-1 -2; -3 -4] := G[-1; -3 1] * H[1 -2; -4]
end

function Base.show(io::IO, scheme::ATRG)
    println(io, "ATRG - Anisotropic TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
