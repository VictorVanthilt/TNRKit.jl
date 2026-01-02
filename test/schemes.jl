# This tests every scheme in the library on the Z2 symmetric Ising model.

println("---------------------")
println(" Testing all schemes ")
println("---------------------")

T = classical_ising_symmetric()
T_3D = classical_ising_symmetric_3D()
# from Fig. 5 of Physical Review B 102, 054432 (2020)
const f_benchmark3D = -3.507

function cft_finalize!(scheme)
    finalize!(scheme)
    return cft_data(scheme)
end

# TRG
@testset "TRG - Ising Model" begin
    @info "TRG ising free energy"
    scheme = TRG(T)
    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 2.0e-6

    @info "TRG ising CFT data"
    scheme = TRG(T)
    run!(scheme, truncrank(24), maxiter(10))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 2.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2
end

# BTRG
@testset "BTRG - Ising Model" begin
    @info "BTRG ising free energy"
    scheme = BTRG(T)
    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 6.0e-8

    @info "BTRG ising CFT data"
    scheme = BTRG(T)
    run!(scheme, truncrank(24), maxiter(10))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 3.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 2.0e-2
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    @info "HOTRG ising free energy"
    scheme = HOTRG(T)
    data = run!(scheme, truncrank(16), maxiter(25))

    @test free_energy(data, ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 6.0e-7

    @info "HOTRG ising CFT data"
    scheme = HOTRG(T)
    run!(scheme, truncrank(16), maxiter(4))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 6.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2
end

# ATRG
@testset "ATRG - Ising Model" begin
    @info "ATRG ising free energy"
    scheme = ATRG(T)
    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(data, ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 3.0e-6

    @info "ATRG ising CFT data"
    scheme = ATRG(T)
    run!(scheme, truncrank(24), maxiter(3))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 1.0e-2
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2
end

# LoopTNR
@testset "LoopTNR - Ising Model" begin
    @info "LoopTNR ising free energy"
    scheme = LoopTNR(T)

    entanglement_criterion = maxiter(100)
    loop_criterion = maxiter(5)

    data = run!(
        scheme, truncrank(8), truncbelow(1.0e-12), maxiter(25), entanglement_criterion,
        loop_criterion
    )

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-6

    @info "LoopTNR ising CFT data"
    scheme = LoopTNR(T)
    run!(scheme, truncrank(12), maxiter(10))

    for shape in [[1, 4, 1], [sqrt(2), 2 * sqrt(2), 0]]
        cft = cft_data(scheme, shape)
        d1, d2 = real(cft[Z2Irrep(1)][1]), real(cft[Z2Irrep(0)][2])
        @info "Obtained lowest scaling dimensions:\n$(d1), $(d2)."
        @test d1 ≈ ising_cft_exact[1] rtol = 5.0e-4
        @test d2 ≈ ising_cft_exact[2] rtol = 5.0e-4
    end

    for shape in [[1, 8, 1], [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]]
        cft = cft_data(scheme, shape, truncrank(12), truncbelow(1.0e-10))
        d1, d2 = real(cft[Z2Irrep(1)][1]), real(cft[Z2Irrep(0)][2])
        @info "Obtained lowest scaling dimensions:\n$(d1), $(d2)."
        @test d1 ≈ ising_cft_exact[1] rtol = 1.0e-3
        @test d2 ≈ ising_cft_exact[2] rtol = 1.0e-3
    end
end

@testset "LoopTNR - Initialization with 2 x 2 unit cell" begin
    loop_criterion = maxiter(5)
    trunc = truncrank(8)
    truncentanglement = truncbelow(1.0e-12)
    entanglement_criterion = maxiter(100)
    scheme = LoopTNR(fill(T, (2, 2)); loop_criterion, trunc, truncentanglement)
    data = run!(
        scheme, truncrank(8), truncbelow(1.0e-12), maxiter(25), entanglement_criterion,
        loop_criterion
    )
    @test free_energy(data, ising_βc; initial_size = 2) ≈ f_onsager rtol = 1.0e-6
end

# SLoopTNR
@testset "SLoopTNR - Ising Model" begin
    @info "SLoopTNR ising free energy"
    T_inv = classical_ising_inv()
    scheme = SLoopTNR(T_inv)

    data = run!(scheme, truncrank(4), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-5
end

# ctm_TRG
@testset "ctm_TRG - Ising Model" begin
    @info "ctm_TRG ising free energy"
    scheme = ctm_TRG(T, 8)
    lz = run!(scheme, truncrank(8), maxiter(25))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 7.0e-6
end

# ctm_HOTRG
@testset "ctm_HOTRG - Ising Model" begin
    @info "ctm_HOTRG ising free energy"
    scheme = ctm_HOTRG(T, 8)
    lz = run!(scheme, truncrank(8), maxiter(25))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 2.0e-5
end

# c4vCTM
@testset "c4vCTM - Ising Model" begin
    @info "c4vCTM ising free energy"
    scheme = c4vCTM(T)
    lz = run!(scheme, truncrank(24), trivial_convcrit(1.0e-9); verbosity = 1)

    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 6.0e-8
end

# rCTM
@testset "rCTM - Ising Model" begin
    @info "rCTM ising free energy"
    scheme = rCTM(T)
    lz = run!(scheme, truncrank(24), trivial_convcrit(1.0e-9); verbosity = 1)

    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 6.0e-8
end

# ATRG_3D
@testset "ATRG_3D - Ising Model" begin
    @info "ATRG_3D ising free energy"
    scheme = ATRG_3D(T_3D)
    data = run!(scheme, truncrank(12), maxiter(25))
    fs = free_energy(data, ising_βc_3D; scalefactor = 8.0)
    @info "Calculated f = $(fs)."
    @test fs ≈ f_benchmark3D rtol = 5.0e-3
end

# HOTRG_3D
@testset "HOTRG_3D - Ising Model" begin
    @info "HOTRG_3D ising free energy"
    scheme = HOTRG_3D(T_3D)
    data = run!(scheme, truncrank(8), maxiter(25))
    fs = free_energy(data, ising_βc_3D; scalefactor = 8.0)
    @info "Calculated f = $(fs)."
    @test fs ≈ f_benchmark3D rtol = 1.0e-3
end

# ImpurityHOTRG
@testset "ImpurityHOTRG - Ising Model" begin

    T = classical_ising()
    T_imp1 = classical_ising_impurity()

    scheme = ImpurityHOTRG(T, T_imp1, T_imp1, T)

    data = run!(scheme, truncrank(16), maxiter(25))

    @test free_energy(getindex.(data, 1), ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 6.0e-7
end

@testset "Impurity HOTRG - Magnetisation" begin
    # High temperature limit (<m^2> -> 0)
    β = 0.2

    T = classical_ising(β)
    T_imp_order1_1 = classical_ising_impurity(β)
    T_imp_order2 = classical_ising(β)

    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)

    data = run!(scheme, truncrank(8), maxiter(25))

    m2_highT = data[end][4] / data[end][1]
    @test m2_highT ≈ 0.0 atol = 1.0e-14

    # Low temperature limit (<m^2> -> 1)
    β = 1.0

    T = classical_ising(β)
    T_imp_order1_1 = classical_ising_impurity(β)
    T_imp_order2 = classical_ising(β)

    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)

    data = run!(scheme, truncrank(8), maxiter(25))

    m2_lowT = data[end][4] / data[end][1]
    @test m2_lowT ≈ 1 rtol = 1.0e-2
end

# ImpurityTRG
@testset "ImpurityTRG - Ising Model" begin
    T = classical_ising()
    T_imp = classical_ising_impurity()

    scheme = ImpurityTRG(T, T_imp, T, T, T)

    data = run!(scheme, truncrank(24), maxiter(25))

    @test free_energy(getindex.(data, 1), ising_βc) ≈ f_onsager rtol = 2.0e-6
end

@testset "ImpurityTRG - Magnetisation" begin
    # High T
    β = 0.1

    T = classical_ising(β)
    T_imp = classical_ising_impurity(β)

    scheme = ImpurityTRG(T, T_imp, T, T, T)

    data = run!(scheme, truncrank(16), maxiter(25))

    m_expection = data[end][2] / data[end][1]
    @test m_expection ≈ 0.0 atol = 1.0e-4

    # Low T
    β = 2

    T = classical_ising(β; h = 1.0e-6)
    T_imp = classical_ising_impurity(β; h = 1.0e-6)

    scheme = ImpurityTRG(T, T_imp, T, T, T)

    data = run!(scheme, truncrank(16), maxiter(25))

    m_expection = data[end][2] / data[end][1]
    @test m_expection ≈ 1.0 rtol = 1.0e-4
end
