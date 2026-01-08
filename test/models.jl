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
    phi4_real(10, -1, 1),
    phi4_real_symmetric(10, -1, 1),
    phi4_complex(6, -1, 1),
    phi4_complex_symmetric(6, -1, 1),
]

temperatures = [
    ising_βc,
    ising_βc,
    1.0,
    potts_βc(3),
    potts_βc(3),
    1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
]

answers = [
    f_onsager, # Hack because classical_ising starts from larger lattice
    f_onsager,
    -1.4515448845652446,
    -4.119552029995684, # This is an approximation!
    -4.119552029995684, # This is an approximation!
    3 / 2 * log(3 / 4),
    3 / 2 * log(3 / 4),
    3 / 2 * log(3 / 4),
    0.4241912271276211, # This is an approximation!
    0.4232381701937374, # This is an approximation!
    0.7583605364656325, # This is an approximation!
    4.91362301061858, # This is an approximation!
]

@testset "2D Models" begin
    for (model, temp, answer) in zip(models_2D, temperatures, answers)
        scheme = TRG(model)
        data = run!(scheme, truncdim(16), maxiter(25))
        @test free_energy(data, temp) ≈ answer rtol = 1.0e-3
    end
end
