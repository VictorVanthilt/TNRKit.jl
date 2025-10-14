const ising_βc_triangular = BigFloat(log(BigFloat(3.0))/BigFloat(4.0))

function classical_ising_symmetric_triangular(β)
    x = cosh(β / 2)
    y = sinh(β / 2)

    S = ℤ₂Space(0 => 1, 1 => 1)
    id_tensor = ones(Float64, S ⊗ S ⊗ S ← S ⊗ S ⊗ S)

    M = zeros(Float64, 2, 2)
    M[1,1] = x
    M[2,2] = y
    M = TensorMap(M,  S ← S)
    @tensor T[-1 -2 -3; -4 -5 -6] := id_tensor[1 2 3; 4 5 6] * M[-1; 1] * M[-2; 2] * M[-3; 3] * M[4; -4] * M[5; -5] * M[6; -6]
    
    return T
end
classical_ising_symmetric_triangular() = classical_ising_symmetric_triangular(ising_βc_triangular)
