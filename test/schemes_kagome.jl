@testset "LoopTNR on Kagome lattice" begin
    T_ising_dual_triangular = classical_ising_dual_triangular()
    TA, TB, TC = honeycomb_to_kagome(T_ising_dual_triangular)
    scheme = KagomeLoopTNR(TA, TB, TC)
    data = run!(scheme, truncdim(8), maxiter(10); max_loop = 20)
    fs = free_energy(data, ising_βc_triangular; scalefactor = 3.0)
    @test fs ≈ f_onsager_triangular rtol = 1.0e-4
end

@testset "LoopTNR on Kagome lattice with Z2 symmetry" begin
    T_ising_dual_triangular = classical_ising_dual_triangular_symmetric()
    TA, TB, TC = honeycomb_to_kagome(T_ising_dual_triangular)
    scheme = KagomeLoopTNR(TA, TB, TC)
    data = run!(scheme, truncdim(10), maxiter(15); max_loop = 20)
    fs = free_energy(data, ising_βc_triangular; scalefactor = 3.0)
    @test fs ≈ f_onsager_triangular rtol = 1.0e-5
end