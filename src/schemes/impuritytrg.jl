"""
$(TYPEDEF)

Impurity method for Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T, T_imp1, T_imp2, T_imp3, T_imp4)

### Running the algorithm
    run!(::ImpurityTRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalizer=ImpurityTRG_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of √2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Morita et. al. Phys. Rev. B 111 (2025)](@cite moritaMultiimpurityMethodBondweighted2025)
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadohTensorNetworkAnalysis2019)
"""
mutable struct ImpurityTRG <: TNRScheme
    T::TensorMap
    T_imp1::TensorMap
    T_imp2::TensorMap
    T_imp3::TensorMap
    T_imp4::TensorMap

    function ImpurityTRG(
            T::TensorMap{E, S, 2, 2}, T_imp1::TensorMap{E, S, 2, 2}, T_imp2::TensorMap{E, S, 2, 2},
            T_imp3::TensorMap{E, S, 2, 2}, T_imp4::TensorMap{E, S, 2, 2}
        ) where {E, S}


        @assert space(T, 1) == space(T_imp1, 1) == space(T_imp2, 1) == space(T_imp3, 1) == space(T_imp4, 1) "First space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        @assert space(T, 2) == space(T_imp1, 2) == space(T_imp2, 2) == space(T_imp3, 2) == space(T_imp4, 2) "Second space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        @assert space(T, 3) == space(T_imp1, 3) == space(T_imp2, 3) == space(T_imp3, 3) == space(T_imp4, 3) "Third space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"
        @assert space(T, 4) == space(T_imp1, 4) == space(T_imp2, 4) == space(T_imp3, 4) == space(T_imp4, 4) "Fourth space of T, T_imp1, T_imp2, T_imp3 and T_imp4 must be the same"

        return new(T, T_imp1, T_imp2, T_imp3, T_imp4)
    end
end

function step!(scheme::ImpurityTRG, trunc::TensorKit.TruncationScheme)
    # Tensor1
    A1, B1 = SVD12(scheme.T_imp1, trunc)

    # Tensor2
    tensor2p = transpose(scheme.T_imp2, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(scheme.T_imp3, trunc)

    # Tensor4
    tensor4p = transpose(scheme.T_imp4, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(scheme.T, trunc)
    tensorpurep = transpose(scheme.T, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar scheme.T[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T_imp1[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T_imp2[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar scheme.T_imp3[-1, -2; -3, -4] := D4[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A3[3 2; -4]
    @planar scheme.T_imp4[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C4[4 3; -3] * A1[3 2; -4]

    return scheme
end

function Base.show(io::IO, scheme::ImpurityTRG)
    println(io, "ImpurityTRG - Impurity TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * T_imp1: $(summary(scheme.T_imp1))")
    println(io, "  * T_imp2: $(summary(scheme.T_imp2))")
    println(io, "  * T_imp3: $(summary(scheme.T_imp3))")
    println(io, "  * T_imp4: $(summary(scheme.T_imp4))")
    return nothing
end

run!(scheme::ImpurityTRG, trscheme::TensorKit.TruncationScheme, criterion::stopcrit) = run!(scheme, trscheme, criterion; finalizer = ImpurityTRG_Finalizer)
