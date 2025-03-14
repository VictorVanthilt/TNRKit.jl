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

    U, S, V, _ = tsvd(scheme.T, ((3, 1), (4, 2)); trunc=trunc)

    @plansor begin
        C[-1 -2; -3] := U[-1 -2; 1] * sqrt(S)[1; -3]
        D[-1; -2 -3] := sqrt(S)[-1; 1] * V[1; -2 -3]
    end

    @tensor scheme.T[-1 -2; -3 -4] := D[-1; 3 1] * B[-2; 1 4] * C[2 4; -4] * A[3 2; -3]
    return scheme
end

# example convcrit function
trg_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

function Base.show(io::IO, scheme::TRG)
    println(io, "TRG - Tensor Renormalization Group")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
