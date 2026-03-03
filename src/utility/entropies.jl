function VN_entropy(M::TensorMap; rtol = 1e-14, power = 1.0)
    _, S, _ = svd_trunc(M; trunc = trunctol(rtol = rtol))
    S_vec_norm = S.data / sum(S.data)
    plogp = map(x -> x > rtol ? x^power * power * log(x) : 0.0, S_vec_norm)
    S_von = - sum(plogp)
    return S_von, S / S.data[1]
end

function loop_entropy(scheme::LoopTNR)
    psi_A = Ψ_A(scheme)
    psi_Apsi_A_vector = ΨAΨA(psi_A)
    N = length(psi_A)
    psi_Apsi_A_cache = right_cache(psi_Apsi_A_vector)

    entropies_circ = Float64[]
    specs_circ = DiagonalTensorMap{Float64}[]
    entropies_rad = Float64[]
    specs_rad = DiagonalTensorMap{Float64}[]

    psi_Apsi_A = psi_Apsi_A_cache[end]

    for i in 1:N
        psi_Apsi_A = psi_Apsi_A * psi_Apsi_A_vector[i]
        transfer = psi_Apsi_A_cache[i] * psi_Apsi_A
        ent_circ, spec_circ = VN_entropy(transfer)
        ent_rad, spec_rad = VN_entropy(transpose(transfer, ((2, 4), (1, 3))))
        push!(entropies_circ, ent_circ)
        push!(specs_circ, spec_circ)
        push!(entropies_rad, ent_rad)
        push!(specs_rad, spec_rad)
    end

    return entropies_circ, specs_circ, entropies_rad, specs_rad
end