potts_βc(q) = log(1.0 + sqrt(q))
function classical_potts(q::Int, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros, V ⊗ V ← V ⊗ V)

    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = -(Int(i == j) + Int(j == l) + Int(l == k) + Int(k == i))
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    return A_potts
end
classical_potts(q::Int) = classical_potts(q, potts_βc(q))

function weyl_heisenberg_matrices(Q::Int, elt = ComplexF64)
    U = zeros(elt, Q, Q) # clock matrix
    V = zeros(elt, Q, Q) # shift matrix
    W = zeros(elt, Q, Q) # DFT
    ω = cis(2 * pi / Q)

    for row in 1:Q
        U[row, row] = ω^(row - 1)
        V[row, mod1(row - 1, Q)] = one(elt)
        for col in 1:Q
            W[row, col] = ω^((row - 1) * (col - 1))
        end
    end
    return U, V, W / sqrt(Q)
end

function classical_potts_symmetric(q::Int64, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = -(Int(i == j) + Int(j == l) + Int(k == l) + Int(k == i))
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    Vp = Vect[ZNIrrep{q}](sector => 1 for sector in 0:(q - 1))
    _, _, W = weyl_heisenberg_matrices(q)
    P = TensorMap(W, ℂ^q ← ℂ^q)
    A_potts = TensorMap(
        reshape(((P' ⊗ P') * A_potts * (P ⊗ P)).data, (q, q, q, q)),
        Vp ⊗ Vp ← Vp ⊗ Vp
    )
    return A_potts
end
classical_potts_symmetric(q::Int) = classical_potts_symmetric(q, potts_βc(q))
