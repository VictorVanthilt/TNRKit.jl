"""
$(SIGNATURES)

Constructs the partition function tensor for the classical clock model with `q` states
and a given inverse temperature `β`.
"""
function classical_clock(q::Int64, β::Float64)
    V = ℂ^q
    A_clock = TensorMap(zeros, V ⊗ V ← V ⊗ V)
    clock(i, j) = -cos(2π / q * (i - j))

    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = clock(i, j) + clock(j, l) + clock(l, k) + clock(k, i)
                    A_clock[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    return A_clock
end

function fourier_matrix(q::Int64, β::Float64)
    U = zeros(ComplexF64, q, q)
    for i in 0:q-1
        for j in 0:q-1
            U[i+1,j+1] = exp(2im*π/q*i*j)/sqrt(q)
        end
    end
    return U = TensorMap(U, ℂ^q←ℂ^q)
end

function classical_clock_symmetric(q::Int64, β::Float64)
    A = classical_clock(q,β)
    U = fourier_matrix(q)
    @tensor Anew[-1 -2;-3 -4] := A[1 2; 3 4] * U[4; -4]*conj(U[1; -1]) *  U[3; -3]*conj(U[2; -2])
    V = ZNSpace{q}(i=>1 for i in 0:q-1)
    return real(TensorMap(Anew[],V⊗V←V⊗V))
end
