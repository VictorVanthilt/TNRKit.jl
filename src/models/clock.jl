"""
$(SIGNATURES)

Constructs the partition function tensor for the classical clock model with `q` states
and a given inverse temperature `β`.
"""
function classical_clock(q::Int64, β::Float64)
    V = ℂ^q
    A_clock = zeros(Float64, V ⊗ V ← V ⊗ V)
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
