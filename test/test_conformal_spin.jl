using TNRKit, TensorKit, KrylovKit
import TNRKit: spec_2x4, next_τ

function finalize_cft_spin!(scheme::LoopTNR; is_real = false)
    T1 = permute(scheme.TA, ((1, 2), (4, 3)))
    T2 = permute(scheme.TB, ((1, 2), (4, 3)))
    n = norm(@tensor opt = true T1[1 2; 3 4] * T2[3 5; 1 6] * T2[7 4; 8 2] * T1[8 6; 7 5])

    scheme.TA /= n^(1 / 4)
    scheme.TB /= n^(1 / 4)
    conformal_data = spec_2x4(scheme.TA, scheme.TB; is_real)
    return conformal_data
end

# simulate 2x1 tensor
O = classical_ising_symmetric()
O2 = block_tensors2([O O;]);
scheme = LoopTNR(O2; finalize = finalize_cft_spin!)
result = run!(scheme, truncdim(12), maxiter(8))

#initial τ
τ = 0.5im
τ_list = []
for i = 1:9
    τ = next_τ(τ)
    push!(τ_list, τ)
end

spec_σ = [result[i][Irrep[ℤ₂](1)][1] for i = 1:9]

x_σ = spec_σ ./ imag(τ_list);
println("x_σ = ", real(x_σ), "\n")

spec_Lm1σ = [result[i][Irrep[ℤ₂](1)][2] for i = 1:9]

# conformal spin of the first descendant. This can be accessed from the odd number of step.
x_Lm1σ = real(spec_σ ./ imag(τ_list))
s_Lm1σ = imag(spec_Lm1σ[1:2:end] ./ real(τ_list[1:2:end]))
println("x_L-1σ = ", x_Lm1σ, "\n")

println("s_L-1σ = ", s_Lm1σ)
