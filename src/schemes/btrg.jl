"""
$(TYPEDEF)

Bond-weighted Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T [, k=-1/2])

### Running the algorithm
    run!(::BTRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Adachi et. al. Phys. Rev. B 105 (2022)](@cite adachiBondweightedTensorRenormalization2022)
"""
mutable struct BTRG{E, S} <: TNRScheme{E, S}
    T::TensorMap{E, S, 2, 2}
    S1::TensorMap{E, S, 1, 1}
    S2::TensorMap{E, S, 1, 1}
    k::E

    function BTRG(T::TensorMap{E, S, 2, 2}, k::Number) where {E, S}
        # Construct S1 and S2 as identity matrices.
        return new{E, S}(T, id(space(T, 2)), id(space(T, 1)), k)
    end
end

# Default implementation using the optimal value for k
BTRG(T::TensorMap) = BTRG(T, -0.5)

function pseudopow(t::DiagonalTensorMap, a::Real; tol = eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for I in LinearAlgebra.diagind(b)
            b[I] = b[I] < tol ? b[I] : b[I]^a
        end
    end
    return t′
end

function step!(scheme::BTRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, ((1, 2), (3, 4)); trunc = trunc)

    S_a = pseudopow(S, (1 - scheme.k) / 2)
    S_b = pseudopow(S, scheme.k)

    @tensor begin
        A[-1 -2; -3] := U[-1 -2; 1] * S_a[1; -3]
        B[-1; -2 -3] := S_a[-1; 1] * V[1; -2 -3]
        S1′[-1; -2] := S_b[-1; -2]
    end

    U, S, V, _ = tsvd(scheme.T, ((3, 1), (4, 2)); trunc = trunc)

    S_a = pseudopow(S, (1 - scheme.k) / 2)
    S_b = pseudopow(S, scheme.k)

    @tensor begin
        C[-1 -2; -3] := U[-1 -2; 1] * S_a[1; -3]
        D[-1; -2 -3] := S_a[-1; 1] * V[1; -2 -3]
        S2′[-1; -2] := S_b[-1; -2]
    end

    @tensor scheme.T[-1 -2; -3 -4] := D[-1; 4 7] *
        scheme.S1[1; 7] *
        B[-2; 1 3] *
        scheme.S2[3; 2] *
        C[8 2; -4] *
        scheme.S1[8; 5] *
        A[6 5; -3] *
        scheme.S2[4; 6]
    scheme.S1 = S1′
    scheme.S2 = S2′
    return scheme
end

function Base.show(io::IO, scheme::BTRG)
    println(io, "BTRG - Bond-weighted TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * S1: $(summary(scheme.S1))")
    println(io, "  * S2: $(summary(scheme.S2))")
    println(io, "  * k: $(scheme.k)")
    return nothing
end
