"""
$(TYPEDEF)

Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T)

### Running the algorithm
    run!(::TRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalizer=default_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step
    
### Fields

$(TYPEDFIELDS)

### References
* [Levin & Nave Phys. Rev. Letters 99(12) (2007)](@cite levinTensorRenormalizationGroup2007)
"""
mutable struct TRG <: TNRScheme
    "central tensor"
    T::TensorMap

    function TRG(T::TensorMap{E, S, 2, 2}) where {E, S}
        return new(T)
    end
end

function step!(scheme::TRG, trunc::TensorKit.TruncationScheme)
    A, B = SVD12(scheme.T, trunc)
    Tp = transpose(scheme.T, ((2, 4), (1, 3)))
    C, D = SVD12(Tp, trunc)
    @plansor scheme.T[-1 -2; -3 -4] := D[-2; 1 2] * B[-1; 4 1] * C[4 3; -3] * A[3 2; -4]
    return scheme
end

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
