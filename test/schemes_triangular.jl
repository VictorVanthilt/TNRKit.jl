# lz_onsager_triangular = 0.8796
# fs_onsager_triangular = -lz_onsager_triangular / ising_βc_triangular  # Approximate value for triangular lattice

# c6CTM
@testset "c6CTM - Ising Model" begin
    for method in [classical_ising_triangular classical_ising_triangular_symmetric]
        T_flipped = method(ising_βc_triangular)

        scheme = c6vCTM_triangular(T_flipped)
        lz = run!(scheme, truncdim(24), maxiter(50))

        fs = lz * -1 / ising_βc_triangular
        @test fs ≈ f_onsager_triangular rtol = 1.0e-3
    end
end
