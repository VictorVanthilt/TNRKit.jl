Potts_βc(q) = log(1.0 + sqrt(q))
function classical_Potts(q::Int, β::Float64)
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
classical_Potts(q::Int) = classical_Potts(q, Potts_βc(q))
