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
    (phi4_real_Z2(10, -1.0, 1.0), -1.0, 0.4232381701937374), # This is an approximation!
    (phi4_complex(6, -1.0, 1.0), -1.0, 0.7583605364656325), # This is an approximation!
    (phi4_complex_U1(6, -1.0, 1.0), -1.0, 0.7673189874157453), # This is an approximation!
    (phi4_complex_Z2Z2(6, -1.0, 1.0), -1.0, 0.7665677554973079), # This is an approximation!

]

@testset "2D Models" begin
    for (model, temp, answer) in model_temp_answer
        scheme = TRG(model)
        data = run!(scheme, truncrank(16), maxiter(25))
        @test free_energy(data, temp) ≈ answer rtol = 1.0e-3
    end
end

@testset "LoopTNR - 2D XY model" begin
    @info "Central charge of KT phase with U(1) symmetry"
    T_KT = classical_XY_U1_symmetric(XY_βc + 0.1, 8)
    scheme = LoopTNR(T_KT)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = cft_data(scheme, [sqrt(2), 2 * sqrt(2), 0])
    central_charge = cft["c"]
    @test central_charge ≈ 1.0 atol = 1.0e-2
    @info "Obtained central charge:\n$central_charge."

    @info "Central charge of symmetric phase with U(1) symmetry"
    T_sym = classical_XY_U1_symmetric(XY_βc - 0.1, 8)
    scheme = LoopTNR(T_sym)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = cft_data(scheme, [sqrt(2), 2 * sqrt(2), 0])
    central_charge = cft["c"]
    @test central_charge ≈ 0.0 atol = 1.0e-13
    @info "Obtained central charge:\n$central_charge."

    @info "Central charge of KT phase with O(2) symmetry"
    T_KT = classical_XY_O2_symmetric(XY_βc + 0.1, 8)
    scheme = LoopTNR(T_KT)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = cft_data(scheme, [sqrt(2), 2 * sqrt(2), 0])
    central_charge = cft["c"]
    @test central_charge ≈ 1.0 atol = 1.0e-2
    @info "Obtained central charge:\n$central_charge."

    @info "Central charge of symmetric phase with O(2) symmetry"
    T_sym = classical_XY_O2_symmetric(XY_βc - 0.1, 8)
    scheme = LoopTNR(T_sym)
    data = run!(scheme, truncrank(16), maxiter(20))
    cft = cft_data(scheme, [sqrt(2), 2 * sqrt(2), 0])
    central_charge = cft["c"]
    @test central_charge ≈ 0.0 atol = 1.0e-13
    @info "Obtained central charge:\n$central_charge."
end
