# c6CTM
@testset "c6CTM - Ising Model" begin
    for sym in [Trivial, Z2Irrep]
        T_flipped = classical_ising_honeycomb(sym, ising_βc_honeycomb; T = ComplexF64)
        println(T_flipped.space)
        scheme = c3vCTM_honeycomb(T_flipped)
        lz = run!(scheme, truncrank(20), convcrit(1.0e-4, (steps, data) -> data) & maxiter(300); verbosity = 1)

        fs = lz * -1 / ising_βc_honeycomb
        @test fs ≈ f_onsager_honeycomb rtol = 1.0e-2
    end
end
