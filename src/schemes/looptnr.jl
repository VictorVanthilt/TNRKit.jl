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
function Ψ_A(scheme::LoopTNR)
    psi = AbstractTensorMap[
        transpose(scheme.TA, ((2,), (1, 3, 4)); copy = true),
        transpose(scheme.TB, ((1,), (3, 4, 2)); copy = true),
        transpose(scheme.TA, ((3,), (4, 2, 1)); copy = true),
        transpose(scheme.TB, ((4,), (2, 1, 3)); copy = true),
    ]
    return psi
end

# Function to construct MPS Ψ_B from MPS Ψ_A. Using a large cut-off dimension in SVD but a small cut-off dimension in loop to increase the precision of initialization.
function Ψ_B(
        ΨA,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
    )
    NA = length(ΨA)
    ΨB = []
    for i in 1:NA
        s1, s2 = SVD12(ΨA[i], truncdim(trunc.dim * 2))
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(10) & convcrit(1.0e-12, ΨB_function)
    in_inds = ones(Int, 2 * NA)
    out_inds = 2 * ones(Int, 2 * NA)
    PR_list, PL_list =
        find_projectors(ΨB, in_inds, out_inds, criterion, trunc & truncentanglement)
    MPO_disentangled!(ΨB, in_inds, out_inds, PR_list, PL_list)
    return ΨB
end

# Construct the list of transfer matrices for ΨAΨA
# ---1'--A--3'---
#       | |
#       1 2
#       | |
# ---2'--A--4'---
function ΨAΨA(psiA)
    NA = length(psiA)
    ΨAΨA_list = []
    for i in 1:NA
        @plansor tmp[-1 -2; -3 -4] := psiA[i][-2; 1 2 -4] * psiA[i]'[1 2 -3; -1]
        push!(ΨAΨA_list, tmp)
    end
    return ΨAΨA_list
end

# Construct the list of transfer matrices for ΨBΨB
# ---1'--B--3'---
#        |
#        1
#        |
# ---2'--B--4'---

function ΨBΨB(psiB)
    NB = length(psiB)
    ΨBΨB_list = []
    for i in 1:NB
        @plansor tmp[-1 -2; -3 -4] := psiB[i][-2; 1 -4] * psiB[i]'[1 -3; -1]
        push!(ΨBΨB_list, tmp)
    end
    return ΨBΨB_list
end

# Construct the list of transfer matrices for ΨBΨA
# ---1'--B-3-B--3'---
#        |   |
#        1   2
#         | |
# ---2'----A----4'---

function ΨBΨA(psiB, psiA)
    NA = length(psiA)
    ΨBΨA_list = []
    for i in 1:NA
        @plansor temp[-1 -2; -3 -4] := psiB[2 * i - 1]'[1 3; -1] * psiA[i][-2; 1 2 -4] *
            psiB[2 * i]'[2 -3; 3]
        push!(ΨBΨA_list, temp)
    end
    return ΨBΨA_list
end

# Function to compute the trace of a list of transfer matrices
function to_number(tensor_list)
    cont = tensor_list[1]
    for tensor in tensor_list[2:end]
        cont = cont * tensor
    end
    return tr(cont)
end

#Entanglement Filtering
entanglement_function(steps, data) = abs(data[end])
entanglement_criterion = maxiter(100) & convcrit(1.0e-15, entanglement_function)

loop_criterion = maxiter(50) & convcrit(1.0e-8, entanglement_function)

# Entanglement filtering function
function entanglement_filtering!(
        scheme::LoopTNR,
        entanglement_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
    )
    ΨA = Ψ_A(scheme)
    PR_list, PL_list =
        find_projectors(ΨA, [1, 1, 1, 1], [3, 3, 3, 3], entanglement_criterion, trunc)

    TA = copy(scheme.TA)
    TB = copy(scheme.TB)

    @plansor scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[4][1; -1] *
        PL_list[1][-2; 2] * PR_list[2][4; -4] *
        PL_list[3][-3; 3]
    @plansor scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[2][-1; 1] *
        PR_list[3][2; -2] * PL_list[4][-4; 4] *
        PR_list[1][3; -3]

    return scheme
end

function entanglement_filtering!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    return entanglement_filtering!(scheme, entanglement_criterion, trunc)
end

# Optimisation functions

# Function to compute the half of the matrix N by inputting the left and right SS transfer matrices
tN(SS_left, SS_right) = SS_right * SS_left

# Function to compute the vector W for a given position in the loop
function tW(pos, psiA, psiB, TSS_left, TSS_right)
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
function opt_T(N, W, psi)
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
        apply_f,
        W,
        psi;
        krylovdim = 20,
        maxiter = 20,
        tol = 1.0e-12,
        verbosity = 0,
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
function right_cache(tensor_list)
    n = length(tensor_list)
    cache = similar(tensor_list)
    cache[end] = id(domain(tensor_list[end]))

    for i in (n - 1):-1:1
        cache[i] = tensor_list[i + 1] * cache[i + 1]
    end

    return cache
end

# A general function to optimize the truncation error of an MPS on a ring.
# Sweeping from left to right, we optimize the tensors in the loop by minimizing the cost function.
# Here cache of right-half-chain is used to minimize the number of multiplications to accelerate the sweeping.
# The transfer matrix on the left is updated after each optimization step.
# The cache technique is from Chenfeng Bao's thesis, see http://hdl.handle.net/10012/14674.
function loop_opt(
        psiA::Array,
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        verbosity::Int,
    )
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

function loop_opt!(
        scheme::LoopTNR,
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        verbosity::Int,
    )
    psiA = Ψ_A(scheme)
    psiB = loop_opt(psiA, loop_criterion, trunc, truncentanglement, verbosity)
    @plansor scheme.TB[-1 -2; -3 -4] := psiB[1][1; 2 -2] * psiB[4][-4; 2 3] *
        psiB[5][3; 4 -3] * psiB[8][-1; 4 1]
    @plansor scheme.TA[-1 -2; -3 -4] := psiB[6][-2; 1 2] * psiB[7][2; 3 -4] *
        psiB[2][-3; 3 4] * psiB[3][4; 1 -1]
    return scheme
end

function step!(
        scheme::LoopTNR,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        entanglement_criterion::stopcrit,
        loop_criterion::stopcrit,
        verbosity::Int,
    )
    entanglement_filtering!(scheme, entanglement_criterion, truncentanglement)
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
        verbosity = 1,
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
                scheme,
                trscheme,
                truncentanglement,
                entanglement_criterion,
                loop_criterion,
                verbosity,
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
        scheme::LoopTNR,
        trscheme::TensorKit.TruncationScheme,
        criterion::stopcrit;
        finalize_beginning = true,
        verbosity = 1,
        max_loop = 50,
        tol_loop = 1.0e-8,
    )
    loop_criterion = maxiter(max_loop) & convcrit(tol_loop, entanglement_function)
    return run!(
        scheme,
        trscheme,
        truncbelow(1.0e-15),
        criterion,
        entanglement_criterion,
        loop_criterion;
        finalize_beginning = finalize_beginning,
        verbosity = verbosity,
    )
end

function Base.show(io::IO, scheme::LoopTNR)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
