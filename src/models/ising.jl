const ising_βc = BigFloat(log(BigFloat(1.0) + sqrt(BigFloat(2.0))) / BigFloat(2.0))
const ising_cft_exact = [
    1 / 8, 1, 9 / 8, 9 / 8, 2, 2, 2, 2, 17 / 8, 17 / 8, 17 / 8, 3, 3,
    3, 3, 3,
    25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8, 25 / 8,
]
function classical_ising(β::Number; h = 0)
    function σ(i::Int64)
        return 2i - 3
    end

    T_array = Float64[
        exp(
                β * (σ(i)σ(j) + σ(j)σ(l) + σ(l)σ(k) + σ(k)σ(i)) +
                h / 2 * β * (σ(i) + σ(j) + σ(k) + σ(l))
            )
            for i in 1:2, j in 1:2, k in 1:2, l in 1:2
    ]

    T = TensorMap(T_array, ℝ^2 ⊗ ℝ^2 ← ℝ^2 ⊗ ℝ^2)

    return T
end
classical_ising() = classical_ising(ising_βc)

function classical_ising_symmetric(β)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(Float64, S ⊗ S ← S ⊗ S)
    block(T, Irrep[ℤ₂](0)) .= [2x^2 2x * y; 2x * y 2y^2]
    block(T, Irrep[ℤ₂](1)) .= [2x * y 2x * y; 2x * y 2x * y]

    return T
end
classical_ising_symmetric() = classical_ising_symmetric(ising_βc)

const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781

function classical_ising_symmetric_3D(β)
    x = cosh(β)
    y = sinh(β)
    W = [sqrt(x) sqrt(y); sqrt(x) -sqrt(y)]
    T_array = zeros(Float64, 2, 2, 2, 2, 2, 2)
    for (i, j, k, l, m, n) in Iterators.product([1:2 for _ in 1:6]...)
        for a in 1:2
            # Outer product of W[a, :] with itself 6 times
            T_array[i, j, k, l, m, n] += W[a, i] * W[a, j] * W[a, k] * W[a, l] * W[a, m] *
                W[a, n]
        end
    end
    S = ℤ₂Space(0 => 1, 1 => 1)
    T = TensorMap(T_array, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    return permute(T, ((1, 4), (5, 6, 2, 3)))
end

function classical_ising_3D(β; J = 1.0)
    K = β * J

    # Boltzmann weights
    t = ComplexF64[exp(K) exp(-K); exp(-K) exp(K)]
    r = eigen(t)
    q = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    # local partition function tensor
    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1
    @tensor o[-1 -2; -3 -4 -5 -6] := O[1 2; 3 4 5 6] * q[-1; 1] * q[-2; 2] * q[-3; 3] *
        q[-4; 4] * q[-5; 5] * q[-6; 6]

    TMS = ℂ^2 ⊗ (ℂ^2)' ← ℂ^2 ⊗ ℂ^2 ⊗ (ℂ^2)' ⊗ (ℂ^2)'

    return TensorMap(o, TMS)
end
const ising_βc_3D = 1.0 / 4.51152469

classical_ising_symmetric_3D() = classical_ising_symmetric_3D(ising_βc_3D)
classical_ising_3D() = classical_ising_3D(ising_βc_3D)
