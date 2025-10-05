println("--------------------")
println(" Testing all models ")
println("--------------------")

models_2D = [
    classical_ising(),
    classical_ising_symmetric(),
    gross_neveu_start(0, 0, 0),
    # classical_clock(), # TODO: find out how to test this model
    classical_potts(3),
    classical_potts_symmetric(3),
    sixvertex(Float64, Trivial),
    sixvertex(Float64, U1Irrep),
    sixvertex(Float64, CU1Irrep),
]

temperatures = [ising_βc, ising_βc, 1.0, potts_βc(3), potts_βc(3), 1.0, 1.0, 1.0]

answers = [
    f_onsager, # Hack because classical_ising starts from larger lattice
    f_onsager,
    -1.4515448845652446,
    -4.119552029995684, # This is an approximation!
    -4.119552029995684, # This is an approximation!
    3 / 2 * log(3 / 4),
    3 / 2 * log(3 / 4),
    3 / 2 * log(3 / 4),
]

lattice_sizes = vcat([2.0], fill(1.0, length(models_2D) - 1))

@testset "2D Models" begin
    for (model, temp, answer, unitcell) in
        zip(models_2D, temperatures, answers, lattice_sizes)
        scheme = TRG(model)
        data = run!(scheme, truncdim(16), maxiter(25))
        @test free_energy(data, temp; initial_size = unitcell) ≈ answer rtol = 1.0e-3
    end
end
