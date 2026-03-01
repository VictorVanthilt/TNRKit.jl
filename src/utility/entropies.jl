function VN_entropy(M::TensorMap; rtol = 1e-14, power = 1.0)
    _, S, _ = svd_trunc(M)
    S_vec_norm = S.data / sum(S.data)
    plogp = map(x -> x > rtol ? x^power * power * log(x) : 0.0, S_vec_norm)
    S_von = - sum(plogp)
    return S_von, S
end