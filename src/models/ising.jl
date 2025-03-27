const ising_βc = BigFloat(log(1.0 + sqrt(2)) / 2.0)
function classical_ising(β::Number; h=0)
    function σ(i::Int64)
        return 2i - 3
    end

    T_array = Float64[exp(β * (σ(i)σ(j) + σ(j)σ(l) + σ(l)σ(k) + σ(k)σ(i)) +
                          h / 2 * β * (σ(i) + σ(j) + σ(k) + σ(l)))
                      for i in 1:2, j in 1:2, k in 1:2, l in 1:2]

    T = TensorMap(T_array, ℝ^2 ⊗ ℝ^2 ← ℝ^2 ⊗ ℝ^2)

    return T
end
classical_ising() = classical_ising(ising_βc)

function classical_ising_symmetric(β)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(Float64, S ⊗ S ← S ⊗ S)
    block(T, Irrep[ℤ₂](0)) .= [2x^2 2x*y; 2x*y 2y^2]
    block(T, Irrep[ℤ₂](1)) .= [2x*y 2x*y; 2x*y 2x*y]

    return T
end
classical_ising_symmetric() = classical_ising_symmetric(ising_βc)

const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781
