"""
$(TYPEDEF)

Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T [, finalize=finalize!])

### Running the algorithm
    run!(::TRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step
    
### Fields

$(TYPEDFIELDS)

### References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levin_tensor_2007)
"""
mutable struct TRG <: TNRScheme
    "central tensor"
    T::TensorMap

    "finalization function"
    finalize!::Function
    function TRG(T::TensorMap{E, S, 2, 2}; finalize = (finalize!)) where {E, S}
        return new(T, finalize)
    end
end

function step!(scheme::TRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, ((1, 2), (3, 4)); trunc = trunc)

    @tensor begin
        A[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        B[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    U, S, V, _ = tsvd(scheme.T, ((3, 1), (4, 2)); trunc = trunc)

    @tensor begin
        C[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        D[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    @tensor scheme.T[-1 -2; -3 -4] := D[-1; 3 1] * B[-2; 1 4] * C[2 4; -4] * A[3 2; -3]
    return scheme
end

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
