"""
$(TYPEDEF)

Single impurity method for Higher-Order Tensor Renormalization Group (for 2nd order)

### Constructors
    $(FUNCTIONNAME)(T, T_imp_order1_1, T_imp_order1_2, T_imp_order2 [, finalize=finalize!])

### Running the algorithm
    run!(::ImpurityHOTRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Morita et al arxiv/1806.10275 (2017)](@cite moritaHigherOrderTensorRenormalization2017)

"""

mutable struct ImpurityHOTRG <: TNRScheme
    T::TensorMap
    T_imp_order1_1::TensorMap
    T_imp_order1_2::TensorMap
    T_imp_order2::TensorMap
    finalize!::Function
    function ImpurityHOTRG(
            T::TensorMap{E, S, 2, 2},
            T_imp_order1_1::TensorMap{E, S, 2, 2},
            T_imp_order1_2::TensorMap{E, S, 2, 2},
            T_imp_order2::TensorMap{E, S, 2, 2},
            ;
            finalize = (finalize!),
        ) where {E, S}

        @assert space(T, 1) == space(T_imp_order1_1, 1) == space(T_imp_order1_2, 1) "First index space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        @assert space(T, 2) == space(T_imp_order1_1, 2) == space(T_imp_order1_2, 2) "Second index space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        @assert space(T, 3) == space(T_imp_order1_1, 3) == space(T_imp_order1_2, 3) "Third index space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        @assert space(T, 4) == space(T_imp_order1_1, 4) == space(T_imp_order1_2, 4) "Fourth index space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        return new(T, T_imp_order1_1, T_imp_order1_2, T_imp_order2, finalize)
    end
end

function step!(scheme::ImpurityHOTRG, trunc::TensorKit.TruncationScheme)
    # join vertically
    @tensor MMdag[-1 -2; -3 -4] :=
        scheme.T[-1 5; 1 2] *
        scheme.T[-2 3; 5 4] *
        conj(scheme.T[-3 6; 1 2]) *
        conj(scheme.T[-4 3; 6 4])

    # get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc = trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc = trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the left, U on the right
    @tensor T[-1 -2; -3 -4] :=
        scheme.T[1 5; -3 3] * conj(U[1 2; -1]) * U[3 4; -4] * scheme.T[2 -2; 5 4]

    #evolve order 1 impurities with the same unitaries
    @tensor T_imp_order1_1[-1 -2; -3 -4] :=
        1 / 2 *
        scheme.T_imp_order1_1[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T[2 -2; 5 4] +
        1 / 2 *
        scheme.T[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T_imp_order1_1[2 -2; 5 4]

    @tensor T_imp_order1_2[-1 -2; -3 -4] :=
        1 / 2 *
        scheme.T_imp_order1_2[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T[2 -2; 5 4] +
        1 / 2 *
        scheme.T[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T_imp_order1_2[2 -2; 5 4]

    #evolve order 2 impurity with the same unitaries and both order 1 impurities
    @tensor T_imp_order2[-1 -2; -3 -4] :=
        1 / 4 *
        scheme.T_imp_order2[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T[2 -2; 5 4] +
        1 / 4 *
        scheme.T[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T_imp_order2[2 -2; 5 4] +
        1 / 4 *
        scheme.T_imp_order1_1[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T_imp_order1_2[2 -2; 5 4] +
        1 / 4 *
        scheme.T_imp_order1_2[1 5; -3 3] *
        conj(U[1 2; -1]) *
        U[3 4; -4] *
        scheme.T_imp_order1_1[2 -2; 5 4]

    scheme.T = T
    scheme.T_imp_order1_1 = T_imp_order1_1
    scheme.T_imp_order1_2 = T_imp_order1_2
    scheme.T_imp_order2 = T_imp_order2

    # join horizontally
    @tensor MMdag[-1 -2; -3 -4] :=
        scheme.T[1 -1; 2 5] *
        scheme.T[5 -2; 4 3] *
        conj(scheme.T[1 -3; 2 6]) *
        conj(scheme.T[6 -4; 4 3])

    # get unitaries
    U, _, _, εₗ = tsvd(MMdag; trunc = trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(MMdag); trunc = trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    # adjoint(U) on the bottom, U on top
    @tensor T[-1 -2; -3 -4] :=
        scheme.T[-1 1; 3 5] * scheme.T[5 2; 4 -4] * conj(U[1 2; -2]) * U[3 4; -3]

    #evolve order 1 impurities with the same unitaries
    @tensor T_imp_order1_1[-1 -2; -3 -4] :=
        1 / 2 *
        scheme.T_imp_order1_1[-1 1; 3 5] *
        scheme.T[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3] +
        1 / 2 *
        scheme.T[-1 1; 3 5] *
        scheme.T_imp_order1_1[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3]

    @tensor T_imp_order1_2[-1 -2; -3 -4] :=
        1 / 2 *
        scheme.T_imp_order1_2[-1 1; 3 5] *
        scheme.T[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3] +
        1 / 2 *
        scheme.T[-1 1; 3 5] *
        scheme.T_imp_order1_2[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3]

    #evolve order 2 impurity with the same unitaries and both order 1 impurities
    @tensor T_imp_order2[-1 -2; -3 -4] :=
        1 / 4 *
        scheme.T_imp_order2[-1 1; 3 5] *
        scheme.T[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3] +
        1 / 4 *
        scheme.T[-1 1; 3 5] *
        scheme.T_imp_order2[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3] +
        1 / 4 *
        scheme.T_imp_order1_1[-1 1; 3 5] *
        scheme.T_imp_order1_2[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3] +
        1 / 4 *
        scheme.T_imp_order1_2[-1 1; 3 5] *
        scheme.T_imp_order1_1[5 2; 4 -4] *
        conj(U[1 2; -2]) *
        U[3 4; -3]

    scheme.T = T
    scheme.T_imp_order1_1 = T_imp_order1_1
    scheme.T_imp_order1_2 = T_imp_order1_2
    scheme.T_imp_order2 = T_imp_order2

    return scheme
end

function Base.show(io::IO, scheme::ImpurityHOTRG)
    println(io, "ImpurityHOTRG - Impurity Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * T_imp_order1_1: $(summary(scheme.T_imp_order1_1))")
    println(io, "  * T_imp_order1_2: $(summary(scheme.T_imp_order1_2))")
    println(io, "  * T_imp_order2: $(summary(scheme.T_imp_order2))")
    return nothing
end
