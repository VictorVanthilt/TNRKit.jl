# @testset "Ising model test of LoopTNR on Kagome lattice" begin
#     T_ising_dual_triangular = classical_ising_dual_triangular()
#     TA, TB, TC = honeycomb_to_kagome(T_ising_dual_triangular)
#     scheme = KagomeLoopTNR(TA, TB, TC)
#     data = run!(scheme, truncdim(8), maxiter(10); max_loop = 20, verbosity = 3)

#     @info "Test free energy"
#     fs = free_energy(data, ising_βc_triangular; scalefactor = 3.0)
#     @test fs ≈ f_onsager_triangular rtol = 1.0e-4
# end

@testset "LoopTNR on Kagome lattice with Z2 symmetry" begin
    T_ising_dual_triangular = classical_ising_dual_triangular_symmetric()
    TA, TB, TC = honeycomb_to_kagome(T_ising_dual_triangular)
    scheme = KagomeLoopTNR(TA, TB, TC)
    # data = run!(scheme, truncdim(5), maxiter(2); max_loop = 3, verbosity = 3)

    # @info "Test free energy"
    # fs = free_energy(data, ising_βc_triangular; scalefactor = 3.0)
    # @test fs ≈ f_onsager_triangular rtol = 1.0e-6

    # @info "Test CFT data"
    @show scheme
    @test isdefined(TNRKit, :cft_data!)
    cft = cft_data!(scheme, [3 / 2, 2 * sqrt(3), 1 / 4])
    d1, d2 = real(cft[Z2Irrep(1)][1]), real(cft[Z2Irrep(0)][2])
    @info "Obtained lowest scaling dimensions:\n$(d1), $(d2)."
    @test d1 ≈ ising_cft_exact[1] rtol = 5.0e-3
    @test d2 ≈ ising_cft_exact[2] rtol = 5.0e-4
end
