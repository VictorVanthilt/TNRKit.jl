"""
$(TYPEDEF)

Simple two-point correlation function for Higher-Order Tensor Renormalization Group

!!! info "Distance"
    Distance is `dist` in 2^{dist} sites apart. E.g. dist=3 means distance 2^3=8 sites apart.

### Constructors
    $(FUNCTIONNAME)(T, T_imp1, T_imp2, dist)

### Running the algorithm
    run!(::CorrelationHOTRG, trunc::TruncationStrategy, niter::Stopcrit[, finalizer=ImpurityHOTRG_Finalizer, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
Piceu J.
"""
mutable struct CorrelationHOTRG{E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
    "Pure Tensor"
    Tpure::TT

    "First type first order impurity tensor (Phase I)"
    T_imp1::Union{TT,Nothing}

    "Second type first order impurity tensor (Phase I)"
    T_imp2::Union{TT,Nothing}

    "The final impurity (Phase II & III)"
    T_imp_final::Union{TT,Nothing}

    "Correlation distance (2^dist)"
    dist::Int

    "Iteration step"
    iter::Int

    function CorrelationHOTRG(
        Tpure::TT,
        Timp1::TT,
        Timp2::TT,
        dist::Int
    ) where {E,S,TT<:AbstractTensorMap{E,S,2,2}}

        @assert dist ≥ 0 "Distance must be non-negative"

        @assert space(Tpure,1) == space(Timp1,1) == space(Timp2,1)
        @assert space(Tpure,2) == space(Timp1,2) == space(Timp2,2)
        @assert space(Tpure,3) == space(Timp1,3) == space(Timp2,3)
        @assert space(Tpure,4) == space(Timp1,4) == space(Timp2,4)

        return CorrelationHOTRG{E,S,TT}(
            Tpure,
            Timp1,
            Timp2,
            nothing,   # no single impurity yet
            dist,
            0
        )
    end
end


"""
    step!(scheme::CorrelationHOTRG, trunc::TensorKit.TruncationScheme)

Perform a single iteration step of the Correlation HOTRG algorithm.

This function progresses through three phases:
- **Phase 1**: Evolve the pure tensor with two separate impurity tensors.
- **Phase 2**: Merge the two impurity tensors into a single impurity tensor.
- **Phase 3**: Continue evolution with the single merged impurity tensor.

The algorithm determines which phase to execute based on the current iteration count.
After each phase, tensors are finalized and the iteration counter is incremented.

# Arguments
- `scheme::CorrelationHOTRG`: The HOTRG scheme containing the tensors and state.
- `trunc::TensorKit.TruncationScheme`: The truncation scheme to apply during tensor operations.

# Returns
- `scheme::CorrelationHOTRG`: The updated scheme with evolved tensors and incremented iteration count.
"""
function step!(
    scheme::CorrelationHOTRG,
    trunc::TensorKit.TruncationScheme
)

    phase = _phase(scheme)

    if phase == 1
        # -----------------------------
        # PHASE 1 — TWO IMPURITIES
        # -----------------------------

        scheme.Tpure, scheme.Timp1 =
            phase1(scheme.Tpure, scheme.Timp1, trunc)

        scheme.Tpure, scheme.Timp2 =
            phase1(scheme.Tpure, scheme.Timp2, trunc)

        scheme.Tpure, scheme.Timp1, scheme.Timp2 =
            finalize_phase1(
                scheme.Tpure,
                scheme.Timp1,
                scheme.Timp2
            )

    elseif phase == 2
        # -----------------------------
        # PHASE 2 — MERGE IMPURITIES
        # -----------------------------

        Tpure_new, Timp_new =
            phase2(
                scheme.Tpure,
                scheme.Timp1,
                scheme.Timp2,
                trunc
            )

        scheme.Tpure, scheme.Timp =
            finalize_phase23(Tpure_new, Timp_new)

        # Explicitly deactivate two-impurity tensors
        scheme.Timp1 = nothing
        scheme.Timp2 = nothing

    else
        # -----------------------------
        # PHASE 3 — SINGLE IMPURITY
        # -----------------------------

        scheme.Tpure, scheme.Timp =
            phase3(
                scheme.Tpure,
                scheme.Timp,
                trunc
            )

        scheme.Tpure, scheme.Timp =
            finalize_phase23(
                scheme.Tpure,
                scheme.Timp
            )
    end

    scheme.iter += 1
    return scheme
end


function run!(scheme::CorrelationHOTRG, trscheme::TruncationStrategy, niter::stopcrit, finalizer::Finalizer{E}; finalize_beginning = true, verbosity = 1) where {E}
    # First check: assert realistic calculation
    @assert niter > dist "niter must be larger than dist"

    data = Vector{E}()

    LoggingExtras.withlevel(; verbosity) do

        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, finalizer.f!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme)
            push!(data, finalizer.f!(scheme))
            steps += 1
            crit = niter(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(niter, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

############################################################################
#                          HELPER FUNCTIONS                                #
############################################################################


function _phase(scheme::CorrelationHOTRG)
    if scheme.iter < scheme.dist
        return 1
    elseif scheme.iter == scheme.dist
        return 2
    else
        return 3
    end
end

"""
    phase1(Tpure, Timp, trunc)

Performs one HOTRG step for phase 1, updating pure and impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp`: Impurity tensor.
- `trunc`: Truncation scheme.

Returns: Updated pure and impurity tensors.
"""
function phase1(Tpure, Timp, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_x(Tpure, Tpure, Uy)
    Timp = 0.5 * (_step_hotrg_x(Timp, Tpure, Uy) + _step_hotrg_x(Tpure, Timp, Uy))
    Tpure = Tpure_temp

    Ux, _ = _get_hotrg_xproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_y(Tpure, Tpure, Ux)
    Timp = 0.5 * (_step_hotrg_y(Timp, Tpure, Ux) + _step_hotrg_y(Tpure, Timp, Ux))
    Tpure = Tpure_temp

    return Tpure, Timp
end

"""
    phase2(Tpure, Timp1, Timp2, trunc)

Performs one HOTRG step for phase 2, combining two impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp1`, `Timp2`: Impurity tensors.
- `trunc`: Truncation scheme.

Returns: Updated pure and impurity tensors.
"""
function phase2(Tpure, Timp1, Timp2, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_x(Tpure, Tpure, Uy)
    Timp = 0.5 * (_step_hotrg_x(Timp1, Timp2, Uy) + _step_hotrg_x(Timp2, Timp1, Uy))
    Tpure = Tpure_temp

    Ux, _ = _get_hotrg_xproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_y(Tpure, Tpure, Ux)
    Timp = 0.5 * (_step_hotrg_y(Timp, Tpure, Ux) + _step_hotrg_y(Tpure, Timp, Ux))
    Tpure = Tpure_temp

    return Tpure, Timp
end

"""
    phase3(Tpure, Timp, trunc)

Performs one HOTRG step for phase 3, updating pure and impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp`: Impurity tensor.
- `trunc`: Truncation scheme.

Returns: Updated pure and impurity tensors.
"""
function phase3(Tpure, Timp, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_x(Tpure, Tpure, Uy)
    Timp = 0.5 * (_step_hotrg_x(Timp, Tpure, Uy) + _step_hotrg_x(Tpure, Timp, Uy))
    Tpure = Tpure_temp

    Ux, _ = _get_hotrg_xproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_y(Tpure, Tpure, Ux)
    Timp = 0.5 * (_step_hotrg_y(Timp, Tpure, Ux) + _step_hotrg_y(Tpure, Timp, Ux))
    Tpure = Tpure_temp

    return Tpure, Timp
end


"""
    finalize_phase1(Tpure, Timp1, Timp2)

Normalizes pure and impurity tensors after phase 1.

Arguments:
- `Tpure`: Pure tensor.
- `Timp1`, `Timp2`: Impurity tensors.

Returns: Normalized pure tensor (twice) and impurity tensors.
"""
function finalize_phase1(Tpure, Timp1, Timp2)
    npure = norm(@tensor Tpure[1 2; 2 1])
    
    Tpure /= npure
    Timp1 /= npure
    Timp2 /= npure

    return Tpure, Tpure, Timp1, Timp2
end

"""
    finalize_phase23(Tpure, Timp)

Normalizes pure and impurity tensors after phases 2 and 3.

Arguments:
- `Tpure`: Pure tensor.
- `Timp`: Impurity tensor.

Returns: Normalized pure and impurity tensors.
"""
function finalize_phase23(Tpure, Timp)
    npure = norm(@tensor Tpure[1 2; 2 1])
    
    Tpure /= npure
    Timp /= npure

    return Tpure, Timp
end
