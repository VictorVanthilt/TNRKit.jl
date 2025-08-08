# This tests every scheme in the library on the Z2 symmetric Ising model.

println("---------------------")
println(" Testing all schemes ")
println("---------------------")

T = classical_ising_symmetric()

function cft_finalize!(scheme)
    finalize!(scheme)
    return cft_data(scheme)
end

# TRG
@testset "TRG - Ising Model" begin
    @info "TRG ising free energy"
    scheme = TRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 2.0e-6

    @info "TRG ising CFT data"
    scheme = TRG(T)
    run!(scheme, truncdim(24), maxiter(10))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 2.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2
end

# BTRG
@testset "BTRG - Ising Model" begin
    @info "BTRG ising free energy"
    scheme = BTRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 6.0e-8

    @info "BTRG ising CFT data"
    scheme = BTRG(T)
    run!(scheme, truncdim(24), maxiter(10))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 3.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 2.0e-2
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    @info "HOTRG ising free energy"
    scheme = HOTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))

    @test free_energy(data, ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 6.0e-7

    @info "HOTRG ising CFT data"
    scheme = HOTRG(T)
    run!(scheme, truncdim(16), maxiter(4))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 6.0e-4
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2
end

# ATRG
@testset "ATRG - Ising Model" begin
    @info "ATRG ising free energy"
    scheme = ATRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    @test free_energy(data, ising_βc; scalefactor = 4.0) ≈ f_onsager rtol = 3.0e-6

    @info "ATRG ising CFT data"
    scheme = ATRG(T)
    run!(scheme, truncdim(24), maxiter(3))

    cft = cft_data(scheme)[2:end]

    @test cft[1] ≈ ising_cft_exact[1] rtol = 1.0e-2
    @test cft[2] ≈ ising_cft_exact[2] rtol = 1.0e-2
end

# LoopTNR
@testset "LoopTNR - Ising Model" begin
    @info "LoopTNR ising free energy"
    scheme = LoopTNR(T)

    entanglement_function(steps, data) = abs(data[end])
    entanglement_criterion = maxiter(100) & convcrit(1.0e-15, entanglement_function)
    loop_criterion = maxiter(5) & convcrit(1.0e-10, entanglement_function)

    data = run!(
        scheme, truncdim(8), truncbelow(1.0e-12), maxiter(25), entanglement_criterion,
        loop_criterion
    )

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-6

    @info "LoopTNR ising CFT data"
    scheme = LoopTNR(T)
    run!(scheme, truncdim(8), maxiter(10))

    for shape in [[1, 4, 1], [sqrt(2), 2 * sqrt(2), 0]]
        cft = cft_data!(scheme, shape)
        @test real(cft[Z2Irrep(1)][1]) ≈ ising_cft_exact[1] rtol = 2.0e-4
        @test real(cft[Z2Irrep(0)][2]) ≈ ising_cft_exact[2] rtol = 1.0e-2
    end

    for shape in [[1, 8, 1], [4 / sqrt(10), 2 * sqrt(10), 2 / sqrt(10)]]
        cft = cft_data!(scheme, shape, truncdim(8), truncbelow(1.0e-10))
        @test real(cft[Z2Irrep(1)][1]) ≈ ising_cft_exact[1] rtol = 1.0e-2
        @test real(cft[Z2Irrep(0)][2]) ≈ ising_cft_exact[2] rtol = 1.0e-2
    end
end

# SLoopTNR
@testset "SLoopTNR - Ising Model" begin
    @info "SLoopTNR ising free energy"
    T_inv = classical_ising_inv()
    scheme = SLoopTNR(T_inv)

    data = run!(scheme, truncdim(4), maxiter(25))

    @test free_energy(data, ising_βc) ≈ f_onsager rtol = 1.0e-5
end

# ctm_TRG
@testset "ctm_TRG - Ising Model" begin
    @info "ctm_TRG ising free energy"
    scheme = ctm_TRG(T, 8)
    lz = run!(scheme, truncdim(8), maxiter(25))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 7.0e-6
end

# ctm_HOTRG
@testset "ctm_HOTRG - Ising Model" begin
    @info "ctm_HOTRG ising free energy"
    scheme = ctm_HOTRG(T, 8)
    lz = run!(scheme, truncdim(8), maxiter(25))
    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 2.0e-5
end

# c4CTM
@testset "c4CTM - Ising Model" begin
    @info "c4CTM ising free energy"
    scheme = c4CTM(T)
    lz = run!(scheme, truncdim(24), trivial_convcrit(1.0e-9); verbosity = 1)

    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 6.0e-8
end

# rCTM
@testset "rCTM - Ising Model" begin
    @info "rCTM ising free energy"
    scheme = rCTM(T)
    lz = run!(scheme, truncdim(24), trivial_convcrit(1.0e-9); verbosity = 1)

    fs = lz * -1 / ising_βc

    @test fs ≈ f_onsager rtol = 6.0e-8
end

# ATRG_3D
@testset "ATRG_3D - Ising Model" begin
    @info "ATRG_3D ising free energy"
    T_3D = classical_ising_symmetric_3D()
    scheme = ATRG_3D(T_3D)
    data = run!(scheme, truncdim(12), maxiter(25))

    fs = free_energy(data, ising_βc_3D; scalefactor = 8.0)

    fs = lnz * -1 / ising_βc_3D
    f_benchmark = -3.515

    @test fs ≈ f_benchmark rtol = 1.0e-3
end
