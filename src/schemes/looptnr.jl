"""
$(TYPEDEF)

Loop Optimization for Tensor Network Renormalization

### Constructors
    $(FUNCTIONNAME)(T [, finalize=finalize!])
    $(FUNCTIONNAME)(TA, TB, [, finalize=finalize!])

### Running the algorithm
    run!(::LoopTNR, trunc::TensorKit.TruncationScheme, truncentanglement::TensorKit.TruncationScheme, criterion::stopcrit,
              entanglement_criterion::stopcrit, loop_criterion::stopcrit[, finalize_beginning=true, verbosity=1])

    run!(::LoopTNR, trscheme::TensorKit.TruncationScheme, criterion::stopcrit[, finalize_beginning=true, verbosity=1])

### Fields

$(TYPEDFIELDS)

### References
* [Yang et. al. Phys. Rev. Letters 118 (2017)](@cite yangLoopOptimizationTensor2017)

"""
mutable struct LoopTNR <: TNRScheme
    TA::TensorMap
    TB::TensorMap

    finalize!::Function
    function LoopTNR(TA::TensorMap, TB::TensorMap; finalize = (finalize!))
        return new(TA, TB, finalize)
    end
    function LoopTNR(T::TensorMap; finalize = (finalize!))
        return new(T, copy(T), finalize)
    end
end

# Function to initialize the list of tensors Ψ_A, making it an MPS on a ring
function Ψ_A(unitcell_2x2::Matrix{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 2, 2}}
    size(unitcell_2x2) == (2, 2) || error("Input unit cell must have 2 x 2 size.")
    ΨA = [
        transpose(unitcell_2x2[1, 1], ((2,), (1, 3, 4)); copy = true),
        transpose(unitcell_2x2[1, 2], ((1,), (3, 4, 2)); copy = true),
        transpose(unitcell_2x2[2, 2], ((3,), (4, 2, 1)); copy = true),
        transpose(unitcell_2x2[2, 1], ((4,), (2, 1, 3)); copy = true),
    ]
    return ΨA
end
function Ψ_A(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    ΨA = [
        transpose(TA, ((2,), (1, 3, 4)); copy = true),
        transpose(TB, ((1,), (3, 4, 2)); copy = true),
        transpose(TA, ((3,), (4, 2, 1)); copy = true),
        transpose(TB, ((4,), (2, 1, 3)); copy = true),
    ]
    return ΨA
end
function Ψ_A(scheme::LoopTNR)
    return Ψ_A(scheme.TA, scheme.TB)
end

# Function to construct MPS Ψ_B from MPS Ψ_A. Using a large cut-off dimension in SVD but a small cut-off dimension in loop to increase the precision of initialization.
function Ψ_B(
        ΨA::Vector{T}, trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme
    ) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 3}}
    NA = length(ΨA)
    ΨB = [s for A in ΨA for s in SVD12(A, truncdim(trunc.dim * 2))]

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1.0e-12, ΨB_function)
    in_inds = ones(Int, 2 * NA)
    out_inds = 2 * ones(Int, 2 * NA)
    PR_list, PL_list = find_projectors(
        ΨB, in_inds, out_inds, criterion,
        trunc & truncentanglement
    )
    MPO_disentangled!(ΨB, in_inds, out_inds, PR_list, PL_list)
    return ΨB
end

# Construct the list of transfer matrices for ΨAΨA
# ---1'--A--3'---
#       | |
#       1 2
#       | |
# ---2'--A--4'---
function ΨAΨA(ΨA::Vector{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 3}}
    return map(ΨA) do A
        return @plansor AA[-1 -2; -3 -4] := A[-2; 1 2 -4] * conj(A[-1; 1 2 -3])
    end
end

# Construct the list of transfer matrices for ΨBΨB
# ---1'--B--3'---
#        |
#        1
#        |
# ---2'--B--4'---
function ΨBΨB(ΨB::Vector{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 2}}
    return map(ΨB) do B
        return @plansor BB[-1 -2; -3 -4] := B[-2; 1 -4] * conj(B[-1; 1 -3])
    end
end

# Construct the list of transfer matrices for ΨBΨA
# ---1'--B-3-B--3'---
#        |   |
#        1   2
#         | |
# ---2'----A----4'---
function ΨBΨA(
        ΨB::Vector{TB}, ΨA::Vector{TA}
    ) where {
        TB <: AbstractTensorMap{<:Any, <:Any, 1, 2},
        TA <: AbstractTensorMap{<:Any, <:Any, 1, 3},
    }
    NA = length(ΨA)
    @assert length(ΨB) == 2 * NA
    return map(1:NA) do i
        return @plansor temp[-1 -2; -3 -4] := conj(ΨB[2 * i - 1][-1; 1 3]) *
            ΨA[i][-2; 1 2 -4] * conj(ΨB[2 * i][3; 2 -3])
    end
end

# Function to compute the trace of a list of transfer matrices
function to_number(tensors::Vector{T}) where {T <: AbstractTensorMap}
    return tr(reduce(*, tensors))
end

#Entanglement Filtering
entanglement_function(steps, data) = abs(data[end])
default_entanglement_criterion = maxiter(100) & convcrit(1.0e-15, entanglement_function)

function _entanglement_filtering(
        TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2},
        entanglement_criterion::stopcrit, trunc::TensorKit.TruncationScheme
    ) where {E, S}
    ΨA = Ψ_A(TA, TB)
    PRs, PLs = find_projectors(
        ΨA, [1, 1, 1, 1], [3, 3, 3, 3],
        entanglement_criterion, trunc
    )
    @plansor TA[-1 -2; -3 -4] := TA[1 2; 3 4] *
        PRs[4][1; -1] * PLs[1][-2; 2] * PRs[2][4; -4] * PLs[3][-3; 3]
    @plansor TB[-1 -2; -3 -4] := TB[1 2; 3 4] *
        PLs[2][-1; 1] * PRs[3][2; -2] * PLs[4][-4; 4] * PRs[1][3; -3]
    return TA, TB
end

# Entanglement filtering function
function entanglement_filtering!(
        scheme::LoopTNR,
        trunc::TensorKit.TruncationScheme,
        entanglement_criterion::stopcrit = default_entanglement_criterion
    )
    scheme.TA, scheme.TB = _entanglement_filtering(
        scheme.TA, scheme.TB, entanglement_criterion, trunc
    )
    return scheme
end

# Optimisation functions

# Function to compute the half of the matrix N by inputting the left and right SS transfer matrices
function tN(SS_left::AbstractTensorMap{E, S, 2, 2}, SS_right::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    return SS_right * SS_left
end

# Function to compute the vector W for a given position in the loop
function tW(
        pos::Int, psiA::Vector{TA}, psiB::Vector{TB},
        TSS_left::AbstractTensorMap{E, S, 2, 2}, TSS_right::AbstractTensorMap{E, S, 2, 2}
    ) where {
        TA <: AbstractTensorMap{<:Any, <:Any, 1, 3},
        TB <: AbstractTensorMap{<:Any, <:Any, 1, 2}, E, S,
    }
    ΨA = psiA[(pos - 1) ÷ 2 + 1]

    tmp = TSS_right * TSS_left

    if iseven(pos)
        ΨB = psiB[pos - 1]
        #--2---ΨB--1'-   --2'---------2--
        #      |       |         |
        #      4       3'        t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @plansor W[-1; -3 -2] := ΨB'[4 -1; 2] * ΨA[3; 4 -3 1] * tmp[-2 1; 2 3]
    else
        ΨB = psiB[pos + 1]
        #-1'--   --2'--ΨB--2----------1'-
        #      |       |         |
        #      3'      4         t
        #       |     |          m
        #        |   |           p
        #         | |            |
        #---3------ΨA------1----------3--
        @plansor W[-1; -3 -2] := ΨB'[4 2; -2] * ΨA[3; -3 4 1] * tmp[2 1; -1 3]
    end

    return W
end

# Function to optimize the tensor T for a given position in the loop by the Krylov method
function opt_T(
        N::AbstractTensorMap{E, S, 2, 2}, W::AbstractTensorMap{E, S, 1, 2},
        psi::AbstractTensorMap{E, S, 1, 2}
    ) where {E, S}
    function apply_f(x::TensorMap)
        #-----1'--   --2'------------1'--
        #          |            |
        #          3'           |
        #          |            N
        #          |            |
        #          |            |
        #---1------x-------2---------1---
        @plansor b[-1; -3 -2] := N[-2 2; -1 1] * x[1; -3 2]
        return b
    end
    new_T, info = linsolve(
        apply_f, W, psi; krylovdim = 20, maxiter = 20, tol = 1.0e-12,
        verbosity = 0
    )
    return new_T
end

# Function to compute the right cache for the transfer matrices. Here we sweep from left to right. At the end we add the identity transfer matrix to the cache.
# cache[1] = T2 * T3 * T4 * T5 * T6 * T7 * T8
# cache[2] = T3 * T4 * T5 * T6 * T7 * T8
# cache[3] = T4 * T5 * T6 * T7 * T8
# ...
# cache[7] = T8
# cache[8] = I
function right_cache(transfer_mats::Vector{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 2, 2}}
    n = length(transfer_mats)
    cache = similar(transfer_mats)
    cache[end] = id(domain(transfer_mats[end]))

    for i in (n - 1):-1:1
        cache[i] = transfer_mats[i + 1] * cache[i + 1]
    end

    return cache
end

# A general function to optimize the truncation error of an MPS on a ring.
# Sweeping from left to right, we optimize the tensors in the loop by minimizing the cost function.
# Here cache of right-half-chain is used to minimize the number of multiplications to accelerate the sweeping.
# The transfer matrix on the left is updated after each optimization step.
# The cache technique is from Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
function loop_opt(
        psiA::Vector{T}, loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme, verbosity::Int
    ) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 3}}
    psiB = Ψ_B(psiA, trunc, truncentanglement)
    NB = length(psiB) # Number of tensors in the MPS Ψ_B
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)
    psiApsiA = ΨAΨA(psiA)
    C = to_number(psiApsiA) # Since C is not changed during the optimization, we can compute it once and use it in the cost function.
    cost = Float64[Inf]
    sweep = 0
    crit = true
    while crit
        right_cache_BB = right_cache(psiBpsiB)
        right_cache_BA = right_cache(psiBpsiA)
        left_BB = id(codomain(psiBpsiB[1])) # Initialize the left transfer matrix for ΨBΨB
        left_BA = id(codomain(psiBpsiA[1])) # Initialize the left transfer matrix for ΨBΨA

        t_start = time()

        if sweep == 0
            tNt = tr(psiBpsiB[1] * right_cache_BB[1])
            tdw = tr(psiBpsiA[1] * right_cache_BA[1])
            wdt = conj(tdw)
            cost_this = real((C + tNt - wdt - tdw) / C)
            if verbosity > 1
                @infov 3 "Initial cost: $cost_this"
            end
            push!(cost, cost_this)
        end

        for pos_psiB in 1:NB
            pos_psiA = (pos_psiB - 1) ÷ 2 + 1 # Position in the MPS Ψ_A

            N = tN(left_BB, right_cache_BB[pos_psiB]) # Compute the half of the matrix N for the current position in the loop, right cache is used to minimize the number of multiplications
            W = tW(pos_psiB, psiA, psiB, left_BA, right_cache_BA[pos_psiA]) # Compute the vector W for the current position in the loop, using the right cache for ΨBΨA

            new_psiB = opt_T(N, W, psiB[pos_psiB]) # Optimize the tensor T for the current position in the loop, with the psiB[pos_psiB] be the initial guess

            psiB[pos_psiB] = new_psiB # Update a single local tensor in the MPS Ψ_B

            @plansor BB_temp[-1 -2; -3 -4] := new_psiB[-2; 1 -4] * new_psiB'[1 -3; -1]
            psiBpsiB[pos_psiB] = BB_temp # Update the transfer matrix for ΨBΨB
            left_BB = left_BB * BB_temp # Update the left transfer matrix for ΨBΨB

            if iseven(pos_psiB) # If the position is even, we also update the transfer matrix for ΨBΨA
                @plansor BA_temp[-1 -2; -3 -4] := psiB[2 * pos_psiA - 1]'[1 3; -1] *
                    psiA[pos_psiA][-2; 1 2 -4] *
                    psiB[2 * pos_psiA]'[2 -3; 3]
                psiBpsiA[pos_psiA] = BA_temp # Update the transfer matrix for ΨBΨA
                left_BA = left_BA * BA_temp # Update the left transfer matrix for ΨBΨA
            end
        end
        sweep += 1

        tNt = tr(left_BB)
        tdw = tr(left_BA)
        wdt = conj(tdw)
        cost_this = real((C + tNt - wdt - tdw) / C)
        push!(cost, cost_this)
        crit = loop_criterion(sweep, cost)
        if verbosity > 1
            @infov 3 "Sweep: $sweep, Cost: $(cost[end]), Time: $(time() - t_start)s" # Included the time taken for the sweep
        end
    end

    return psiB
end

function ΨB_to_TATB(psiB::Vector{T}) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 2}}
    @plansor TA[-1 -2; -3 -4] := psiB[6][-2; 1 2] * psiB[7][2; 3 -4] *
        psiB[2][-3; 3 4] * psiB[3][4; 1 -1]
    @plansor TB[-1 -2; -3 -4] := psiB[1][1; 2 -2] * psiB[4][-4; 2 3] *
        psiB[5][3; 4 -3] * psiB[8][-1; 4 1]
    return TA, TB
end

function loop_opt!(
        scheme::LoopTNR,
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        verbosity::Int
    )
    psiA = Ψ_A(scheme)
    psiB = loop_opt(psiA, loop_criterion, trunc, truncentanglement, verbosity)
    scheme.TA, scheme.TB = ΨB_to_TATB(psiB)
    return scheme
end

function step!(
        scheme::LoopTNR,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        entanglement_criterion::stopcrit,
        loop_criterion::stopcrit,
        verbosity::Int
    )
    entanglement_filtering!(scheme, truncentanglement, entanglement_criterion)
    loop_opt!(scheme, loop_criterion, trunc, truncentanglement, verbosity::Int)
    return scheme
end

function run!(
        scheme::LoopTNR,
        trscheme::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        criterion::stopcrit,
        entanglement_criterion::stopcrit,
        loop_criterion::stopcrit;
        finalize_beginning = true,
        verbosity = 1
    )
    data = []

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, scheme.finalize!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(
                scheme, trscheme, truncentanglement, entanglement_criterion,
                loop_criterion, verbosity
            )
            push!(data, scheme.finalize!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end


function run!(
        scheme::LoopTNR, trscheme::TensorKit.TruncationScheme, criterion::stopcrit;
        finalize_beginning = true, verbosity = 1, max_loop = 50, tol_loop = 1.0e-8
    )
    loop_criterion = maxiter(max_loop) & convcrit(tol_loop, entanglement_function)
    return run!(
        scheme, trscheme, truncbelow(1.0e-15), criterion, default_entanglement_criterion,
        loop_criterion;
        finalize_beginning = finalize_beginning,
        verbosity = verbosity
    )
end

function Base.show(io::IO, scheme::LoopTNR)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
