# c6CTM
@testset "c6CTM - Ising Model" begin
    for method in [classical_ising_triangular classical_ising_triangular_symmetric]
        T_flipped = method(ising_βc_triangular)

        scheme = c6vCTM_triangular(T_flipped)
        lz = run!(scheme, truncdim(20), maxiter(50))

        fs = lz * -1 / ising_βc_triangular
        @test fs ≈ f_onsager_triangular rtol = 1.0e-4
    end
end
