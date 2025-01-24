mutable struct SLoopTNR <: TRGScheme
    T::TensorMap

    optimization_algorithm::OptimKit.OptimizationAlgorithm
    finalize!::Function
    function SLoopTNR(T::TensorMap; optimization_algorithm=LBFGS(8; verbosity=1, maxiter=500, gradtol=1e-4), finalize=finalize!)
        @assert scalartype(T) <: Real "SLoopTNR only supports real-valued TensorMaps"
        return new(T, optimization_algorithm, finalize)
    end
end

function step!(scheme::SLoopTNR, trunc::TensorKit.TruncationScheme)
    f(A) = _SLoopTNR_cost(permute(scheme.T, ((1, 2), (4, 3))), A) # Another convention was used when implementing SLoopTNR

    function fg(f, A)
        f, g = Zygote.withgradient(f, A)
        return f, g[1]
    end

    Zygote.refresh()

    U, S, _ = tsvd(permute(scheme.T, ((1, 2), (4, 3))); trunc=trunc)
    S₀ = U * sqrt(S)
    if norm(imag(S)) > 1e-12
        @error "S is not real"
    end
    S_opt, _, _, _, _ = optimize(A -> fg(f, A), S₀, scheme.optimization_algorithm)

    @tensor scheme.T[-1 -2; -4 -3] := S_opt[1 2 -3] * S_opt[1 4 -1] * S_opt[3 4 -2] *
                                      S_opt[3 2 -4]
end

function ψAψA(T::AbstractTensorMap)
    @tensor M[-1 -2 -3 -4] := T[1 -2 2 -4] * conj(T[1 -1 2 -3])
    @tensor MM[-1 -2 -3 -4] := M[-1 -2 1 2] * M[-3 -4 1 2]
    return @tensor MM[1 2 3 4] * MM[1 2 3 4]
end

function ψAψB(T::AbstractTensorMap, S::AbstractTensorMap)
    @tensor M[-1 -2 -3 -4] := T[1 -2 2 -4] * conj(S[1 -1 3]) * conj(S[2 -3 3])
    @tensor MM[-1 -2 -3 -4] := M[-1 -2 1 2] * M[-3 -4 1 2]
    @tensor result = MM[1 2 3 4] * MM[1 2 3 4]
    if norm(imag(result)) > 1e-12
        @error "We only support real tensors"
    end
    return result
end

function ψBψB(S::AbstractTensorMap)
    @tensor M[-1 -2 -3 -4] := S[1 -1 3] * conj(S[1 -2 4]) * S[2 -3 3] * conj(S[2 -4 4])
    @tensor MM[-1 -2 -3 -4] := M[-1 -2 1 2] * M[-3 -4 1 2]
    return @tensor MM[1 2 3 4] * MM[1 2 3 4] # This seems very bad for complex numbers
end

function _SLoopTNR_cost(T::AbstractTensorMap, S::AbstractTensorMap)
    return ψAψA(T) - 2 * real(ψAψB(T, S)) + ψBψB(S)
end

slooptnr_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::SLoopTNR)
    println(io, "SLoopTNR - Symmetric Loop TNR")
    println(io, "  * T: $(summary(scheme.T))")
    return println(io,
                   "  * Optimization algorithm: $(summary(scheme.optimization_algorithm))")
end