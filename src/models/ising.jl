const Ising_βc = BigFloat(log(1.0 + sqrt(2)) / 2.0)
Potts_βc(q) = log(1.0 + sqrt(q))

const f_onsager::BigFloat = -2.10965114460820745966777928351108478082549327543540531781696107967700291143188081390114126499095041781

function classical_ising(β::Number; h=0)
    function σ(i::Int64)
        return 2i - 3
    end

    T_array = Float64[exp(β * (σ(i)σ(j) + σ(j)σ(k) + σ(k)σ(l) + σ(l)σ(i)) +
                          h / 2 * β * (σ(i) + σ(j) + σ(k) + σ(l)))
                      for i in 1:2, j in 1:2, k in 1:2, l in 1:2]

    T = TensorMap(T_array, ℝ^2 ⊗ ℝ^2 ← ℝ^2 ⊗ ℝ^2)

    return T
end

function classical_ising_symmetric(β)
    V = Vect[Z2Irrep](0 => 1, 1 => 1)
    Ising = zeros(2, 2, 2, 2)
    c = cosh(β)
    s = sinh(β)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                for l in 1:2
                    if (i + j + k + l) == 4
                        Ising[i, j, k, l] = 2 * c * c
                    elseif (i + j + k + l) == 6
                        Ising[i, j, k, l] = 2 * c * s
                    elseif (i + j + k + l) == 8
                        Ising[i, j, k, l] = 2 * s * s
                    end
                end
            end
        end
    end
    return TensorMap(Ising, V ⊗ V ← V ⊗ V)
end

function classical_Potts(q::Int64, β::Float64)
    V = ℂ^q
    A_potts = TensorMap(zeros, V ⊗ V ← V ⊗ V)

    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = -(Int(i == j) + Int(j == k) + Int(k == l) + Int(l == i))
                    A_potts[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    return A_potts
end

function classical_Clock(q::Int64, β::Float64)
    V = ℂ^q
    A_clock = TensorMap(zeros, V ⊗ V ← V ⊗ V)
    clock(i, j) = -cos(2π / q * (i - j))

    for i in 1:q
        for j in 1:q
            for k in 1:q
                for l in 1:q
                    E = clock(i, j) + clock(j, k) + clock(k, l) + clock(l, i)
                    A_clock[i, j, k, l] = exp(-β * E)
                end
            end
        end
    end
    return A_clock
end
