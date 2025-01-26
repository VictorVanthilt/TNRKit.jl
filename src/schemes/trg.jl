mutable struct TRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function TRG(T::TensorMap; finalize=finalize!)
        return new(T, finalize)
    end
end

function step!(scheme::TRG, trunc::TensorKit.TruncationScheme)
    U, S, V, _ = tsvd(scheme.T, ((1, 2), (3, 4)); trunc=trunc)

    @plansor begin
        A[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        B[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    U, S, V, _ = tsvd(scheme.T, ((1, 4), (2, 3)); trunc=trunc)

    # Flip legs to their original domain (to mitigate space mismatch at the end)
    U = permute(U, ((1,), (2, 3)))
    V = permute(V, ((1, 2), (3,)))

    @plansor begin
        C[-1; -2 -3] := U[-1; -2 1] * sqrt(S)[1; -3]
        D[-1 -2; -3] := sqrt(S)[-1; 1] * V[1 -2; -3]
    end

    # @plansor complains here, not sure why
    @tensor scheme.T[-1 -2; -3 -4] := D[-1 1; 4] * B[-2; 2 1] * C[2; 3 -3] * A[4 3; -4]
    return scheme
end

# example convcrit function
trg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
