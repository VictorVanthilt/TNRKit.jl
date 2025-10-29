"""
$(TYPEDEF)

Single impurity method for Higher-Order Tensor Renormalization Group (for 2nd order)

### Constructors
    $(FUNCTIONNAME)(T, T_imp_order1_1, T_imp_order1_2, T_imp_order2)

### Running the algorithm
    run!(::ImpurityHOTRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalizer=ImpurityHOTRG_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Morita et al 10.1016/j.cpc.2018.10.014 (2018)](@cite moritaCalculationHigherorderMoments2019)

"""
mutable struct ImpurityHOTRG <: TNRScheme
    T::TensorMap
    T_imp_order1_1::TensorMap
    T_imp_order1_2::TensorMap
    T_imp_order2::TensorMap

    function ImpurityHOTRG(
            T::TensorMap{E, S, 2, 2},
            T_imp_order1_1::TensorMap{E, S, 2, 2},
            T_imp_order1_2::TensorMap{E, S, 2, 2},
            T_imp_order2::TensorMap{E, S, 2, 2},
        ) where {E, S}

        @assert space(T, 1) == space(T_imp_order1_1, 1) == space(T_imp_order1_2, 1) "First space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        @assert space(T, 2) == space(T_imp_order1_1, 2) == space(T_imp_order1_2, 2) "Second space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        @assert space(T, 3) == space(T_imp_order1_1, 3) == space(T_imp_order1_2, 3) "Third space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        @assert space(T, 4) == space(T_imp_order1_1, 4) == space(T_imp_order1_2, 4) "Fourth space of T, T_imp_order1_1 and T_imp_order1_2 must be the same"
        return new(T, T_imp_order1_1, T_imp_order1_2, T_imp_order2)
    end
end

function step!(scheme::ImpurityHOTRG, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(scheme.T, scheme.T, trunc)

    T = _step_hotrg_x(scheme.T, scheme.T, Uy)
    T_imp_order1_1 = 0.5 * (_step_hotrg_x(scheme.T_imp_order1_1, scheme.T, Uy) + _step_hotrg_x(scheme.T, scheme.T_imp_order1_1, Uy))
    T_imp_order1_2 = 0.5 * (_step_hotrg_x(scheme.T_imp_order1_2, scheme.T, Uy) + _step_hotrg_x(scheme.T, scheme.T_imp_order1_2, Uy))
    T_imp_order2 = 0.25 * (
        _step_hotrg_x(scheme.T_imp_order2, scheme.T, Uy) +
            _step_hotrg_x(scheme.T, scheme.T_imp_order2, Uy) +
            _step_hotrg_x(scheme.T_imp_order1_1, scheme.T_imp_order1_2, Uy) +
            _step_hotrg_x(scheme.T_imp_order1_2, scheme.T_imp_order1_1, Uy)
    )

    scheme.T = T
    scheme.T_imp_order1_1 = T_imp_order1_1
    scheme.T_imp_order1_2 = T_imp_order1_2
    scheme.T_imp_order2 = T_imp_order2

    Ux, _ = _get_hotrg_xproj(scheme.T, scheme.T, trunc)

    T = _step_hotrg_y(scheme.T, scheme.T, Ux)
    T_imp_order1_1 = 0.5 * (_step_hotrg_y(scheme.T_imp_order1_1, scheme.T, Ux) + _step_hotrg_y(scheme.T, scheme.T_imp_order1_1, Ux))
    T_imp_order1_2 = 0.5 * (_step_hotrg_y(scheme.T_imp_order1_2, scheme.T, Ux) + _step_hotrg_y(scheme.T, scheme.T_imp_order1_2, Ux))
    T_imp_order2 = 0.25 * (
        _step_hotrg_y(scheme.T_imp_order2, scheme.T, Ux) +
            _step_hotrg_y(scheme.T, scheme.T_imp_order2, Ux) +
            _step_hotrg_y(scheme.T_imp_order1_1, scheme.T_imp_order1_2, Ux) +
            _step_hotrg_y(scheme.T_imp_order1_2, scheme.T_imp_order1_1, Ux)
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

run!(scheme::ImpurityHOTRG, trscheme::TensorKit.TruncationScheme, criterion::stopcrit) = run!(scheme, trscheme, criterion; finalizer = ImpurityHOTRG_Finalizer)
