mutable struct GILTTNR <: TNRScheme
    T::TensorMap

    ε::Float64
    giltcrit::stopcrit
    finalize!::Function
    function GILTTNR(T::TensorMap; ε=5e-8, giltcrit=maxiter(20), finalize=finalize!)
        return new(copy(T), ε, giltcrit, finalize)
    end
end

function step!(scheme::GILTTNR, trunc::TensorKit.TruncationScheme)
    # step 1: GILT
    giltscheme = GILT(scheme.T; ε=scheme.ε)

    gilt_steps = 0
    crit = true
    ns = (0.0, 0.0, 0.0, 0.0)

    @infov 3 "Starting GILT\n$(giltscheme)\n"
    t = @elapsed while crit
        _, ns = _step!(giltscheme, truncbelow(scheme.ε))

        gilt_steps += 1
        
        crit = scheme.giltcrit(gilt_steps, maximum(ns))
        @infov 4 "GILT step $gilt_steps, norms: $ns"
    end

    @infov 3 "GILT finished\n $(stopping_info(scheme.giltcrit, gilt_steps, ns))\n Elapsed time: $(t)s\n Iterations: $gilt_steps"

    U, S, V, _ = tsvd(giltscheme.T1, ((1, 2), (3, 4)); trunc=trunc)

    @plansor begin
        A[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        B[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    U, S, V, _ = tsvd(giltscheme.T2, ((1, 4), (2, 3)); trunc=trunc)

    # Flip legs to their original domain (to mitigate space mismatch at the end)
    U = permute(U, ((1,), (2, 3)))
    V = permute(V, ((1, 2), (3,)))

    @plansor begin
        C[-1; -2 -3] := U[-1; -2 1] * sqrt(S)[1; -3]
        D[-1 -2; -3] := sqrt(S)[-1; 1] * V[1 -2; -3]
    end

    @tensor scheme.T[-1 -2; -3 -4] := D[-1 1; 4] * B[-2; 3 1] * C[3; 2 -3] * A[4 2; -4]
    return scheme
end

gilttnr_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::GILTTNR)
    println(io, "Gilt-TRN - GILT + TRG")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * ε: $(scheme.ε)")
    return nothing
end
