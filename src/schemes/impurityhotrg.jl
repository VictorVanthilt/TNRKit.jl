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
* [Morita et al arxiv/1806.10275 (2018)](@cite moritaHigherOrderTensorRenormalization2017)

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

function _step_impurityhotrg_x(
        A1::TensorMap{E, S, 2, 2}, A2::TensorMap{E, S, 2, 2},
        U::TensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the x-direction
                -3
                |
            ┌3--U--4┐
            |       |
        -1--A1--5---A2-- -4
            |       |
            └1--U†-2┘
                |
                -2
    =#

    @tensor T[-1 -2; -3 -4] :=
        A1[-1 1; 3 5] * A2[5 2; 4 -4] * conj(U[1 2; -2]) * U[3 4; -3]
    return T
end

function _step_impurityhotrg_y(
        A1::TensorMap{E, S, 2, 2}, A2::TensorMap{E, S, 2, 2},
        U::TensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the y-direction
                    -3
                    |
            ┌---1---A2---3--┐
            |       |       |
        -1--U†      5       U-- -4
            |       |       |
            └---2---A1---4--┘
                    |
                    -2
    =#

    @tensor T[-1 -2; -3 -4] :=
        conj(U[1 2; -1]) * U[3 4; -4] * A2[1 5; -3 3] * A1[2 -2; 5 4]
    return T
end

function step!(scheme::ImpurityHOTRG, trunc::TensorKit.TruncationScheme)
    U, _ = _get_hotrg_xproj(scheme.T, scheme.T, trunc)
    T = _step_hotrg_x(scheme.T, scheme.T, trunc)
    T_imp_order1_1 = 0.5 * (_step_impurityhotrg_x(scheme.T_imp_order1_1, scheme.T, U) + _step_impurityhotrg_x(scheme.T, scheme.T_imp_order1_1, U))
    T_imp_order1_2 = 0.5 * (_step_impurityhotrg_x(scheme.T_imp_order1_2, scheme.T, U) + _step_impurityhotrg_x(scheme.T, scheme.T_imp_order1_2, U))
    T_imp_order2 = 0.25 * (
        _step_impurityhotrg_x(scheme.T_imp_order2, scheme.T, U) +
            _step_impurityhotrg_x(scheme.T, scheme.T_imp_order2, U) +
            _step_impurityhotrg_x(scheme.T_imp_order1_1, scheme.T_imp_order1_2, U) +
            _step_impurityhotrg_x(scheme.T_imp_order1_2, scheme.T_imp_order1_1, U)
    )
    scheme.T = T
    scheme.T_imp_order1_1 = T_imp_order1_1
    scheme.T_imp_order1_2 = T_imp_order1_2
    scheme.T_imp_order2 = T_imp_order2
    U, _ = _get_hotrg_yproj(scheme.T, scheme.T, trunc)
    T = _step_hotrg_y(scheme.T, scheme.T, trunc)
    T_imp_order1_1 = 0.5 * (_step_impurityhotrg_y(scheme.T_imp_order1_1, scheme.T, U) + _step_impurityhotrg_y(scheme.T, scheme.T_imp_order1_1, U))
    T_imp_order1_2 = 0.5 * (_step_impurityhotrg_y(scheme.T_imp_order1_2, scheme.T, U) + _step_impurityhotrg_y(scheme.T, scheme.T_imp_order1_2, U))
    T_imp_order2 = 0.25 * (
        _step_impurityhotrg_y(scheme.T_imp_order2, scheme.T, U) +
            _step_impurityhotrg_y(scheme.T, scheme.T_imp_order2, U) +
            _step_impurityhotrg_y(scheme.T_imp_order1_1, scheme.T_imp_order1_2, U) +
            _step_impurityhotrg_y(scheme.T_imp_order1_2, scheme.T_imp_order1_1, U)
    )
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
