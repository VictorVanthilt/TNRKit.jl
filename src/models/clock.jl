function clock_tensor(::Type{Trivial}, q::Int, β::Real)
    V = ℂ^q
    A_clock = zeros(Float64, V ⊗ V ← V ⊗ V)
    clock(i, j) = -cos(2π / q * (i - j))

    for i in 1:q, j in 1:q, k in 1:q, l in 1:q
        E = clock(i, j) + clock(j, l) + clock(l, k) + clock(k, i)
        A_clock[i, j, k, l] = exp(-β * E)
    end

    return A_clock
end

"""
    classical_clock(S::Type{Trivial}, q::Int, β::Real)
    classical_clock(::Type{ZNIrrep{N}}, q::Int, β::Real) where {N}

Constructs the partition function tensor for the classical clock model with `q` states
and a given inverse temperature `β`.

Compatible with no symmetry or with explicit ℤq symmetry on each of its spaces.
Defaults to ℤq symmetry if `S` is not provided.
"""
function classical_clock(q::Int, β::Real)
    return classical_clock(ZNIrrep{q}, q, β)
end
function classical_clock(S::Type{Trivial}, q::Int, β::Real)
    return clock_tensor(S, q, β)
end
function classical_clock(::Type{ZNIrrep{N}}, q::Int, β::Real) where {N}
    @assert N == q "number of irreps must match the number of states"
    A = classical_clock(Trivial, q, β)

    # Construct the Fourier matrix for the clock model
    Udat = zeros(ComplexF64, q, q)
    for i in 0:(q - 1)
        for j in 0:(q - 1)
            Udat[i + 1, j + 1] = cispi(2 / q * i * j) / sqrt(q)
        end
    end
    U = TensorMap(Udat, ℂ^q ← ℂ^q)

    @tensor Anew[-1 -2;-3 -4] := A[1 2; 3 4] * U[4; -4] * conj(U[1; -1]) * U[3; -3] * conj(U[2; -2])
    V = ZNSpace{q}(i => 1 for i in 0:(q - 1))
    return real(TensorMap(convert(Array, Anew), V ⊗ V ← V ⊗ V))
end
