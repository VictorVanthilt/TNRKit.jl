#! format: off
function P_tensor()
    P = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
        P[pi1+1, pj1+1, pi2+1, pj2+1, i1+1, j1+1, i2+1, j2+1] =
            i1 * (j1 + j2 + pi1 + pi2) + i2 * (j2 + pi1 + pi2) + pj1 * (pi1 + pi2) + pj2 * pi2 + pi1 + pi2
    end
    return P
end

# Manually defining the A and A_bar tensors
A = zeros(ComplexF64, 2, 2, 2, 2)
A[2, 2, 1, 1] = 1 + 1im
A[1, 1, 2, 2] = -1 - 1im
A[2, 1, 1, 2] = 1 - 1im
A[2, 1, 2, 1] = 2
A[1, 2, 1, 2] = -2im
A[1, 2, 2, 1] = 1 - 1im

A_bar = zeros(ComplexF64, 2, 2, 2, 2)
A_bar[2, 2, 1, 1] = -1 + 1im
A_bar[1, 1, 2, 2] = 1 - 1im
A_bar[2, 1, 1, 2] = -1 - 1im
A_bar[2, 1, 2, 1] = -2
A_bar[1, 2, 1, 2] = -2im
A_bar[1, 2, 2, 1] = -1 - 1im

function gross_neveu_8_leg_tensor(μ::Number, m::Number, g::Number)
    # Utility Kronecker delta function
    δ(x, y) = ==(x, y)

    T = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)
    P = P_tensor()
    V = Vect[FermionParity](0 => 1, 1 => 1)
    for (pi1, pj1, pi2, pj2, i1, j1, i2, j2) in Iterators.product([0:1 for _ in 1:8]...)
        p = P[pi1+1, pj1+1, pi2+1, pj2+1, i1+1, j1+1, i2+1, j2+1]
        T[pi1+1, pj1+1, pi2+1, pj2+1, i2+1, j2+1, i1+1, j1+1] =
            ((-1)^p) * exp(0.5 * μ * (i2 - j2 + pi2 - pj2)) * ((1 / sqrt(2))^(i1 + i2 + j1 + j2 + pi1 + pi2 + pj1 + pj2)) *
            (((m + 2)^2 + 2 * g^2) * δ(i1 + i2 + pj1 + pj2, 0) * δ(j1 + j2 + pi1 + pi2, 0) -
             (m + 2) * δ(i1 + i2 + pj1 + pj2, 1) * δ(j1 + j2 + pi1 + pi2, 1) -
             ((-1)^(i1 + i2 + j2 + pi1)) * (1im^(i2 + j2 + pi2 + pj2)) * (m + 2) * δ(i1 + i2 + pj1 + pj2, 1) * δ(j1 + j2 + pi1 + pi2, 1) -
             A_bar[i1+1, i2+1, pj1+1, pj2+1] * A[j1+1, j2+1, pi1+1, pi2+1])

    end
    return TensorMap(T, V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V)
end
#! format: on

function gross_neveu_start(μ::Number, m::Number, g::Number)
    T_unfused = gross_neveu_8_leg_tensor(μ, m, g)
    V = Vect[FermionParity](0 => 1, 1 => 1)
    U = isometry(fuse(V, V), V ⊗ V)
    Udg = adjoint(U)

    @tensor T_fused[-1 -2; -3 -4] := T_unfused[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] *
                                     Udg[5 6; -3] * Udg[7 8; -4]

    # restore the TNRKit.jl convention
    return permute(T_fused, ((1, 2), (4, 3)))
end
