println("--------------------")
println(" Testing all models ")
println("--------------------")

model_temp_answer = [
    (classical_ising(), ising_βc, f_onsager),
    (classical_ising_symmetric(), ising_βc, f_onsager),
    (gross_neveu_start(0, 0, 0), 1.0, -1.4515448845652446),
    (classical_clock(3, 2.0 * log(√3 + 1) / 3), 2.0 * log(√3 + 1) / 3, -4.17924244901635), # This is an approximation!
    (classical_clock_symmetric(3, 2.0 * log(√3 + 1) / 3), 2.0 * log(√3 + 1) / 3, -4.17924244901635), # This is an approximation!
    (classical_potts(3), potts_βc(3), -4.119552029995684), # This is an approximation!
    (classical_potts_symmetric(3), potts_βc(3), -4.119552029995684), # This is an approximation!
    (sixvertex(Float64, Trivial), 1.0, 3 / 2 * log(3 / 4)),
    (sixvertex(Float64, U1Irrep), 1.0, 3 / 2 * log(3 / 4)),
    (sixvertex(Float64, CU1Irrep), 1.0, 3 / 2 * log(3 / 4)),
    (phi4_real(10, -1.0, 1.0), -1.0, 0.4241912271276211), # This is an approximation!
    (phi4_real_symmetric(10, -1.0, 1.0), -1.0, 0.4232381701937374), # This is an approximation!
    (phi4_complex(6, -1.0, 1.0), -1.0, 0.7583605364656325), # This is an approximation!
    (phi4_complex_symmetric(6, -1.0, 1.0), -1.0, 4.91362301061858), # This is an approximation!
]

@testset "2D Models" begin
    for (model, temp, answer) in model_temp_answer
        scheme = TRG(model)
        data = run!(scheme, truncdim(16), maxiter(25))
        @test free_energy(data, temp) ≈ answer rtol = 1.0e-3
    end
end
