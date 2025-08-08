f_bench = -1.4515448845652446
T = gross_neveu_start(0, 0, 0)

# === TRG ===
@testset "TRG - Gross-Neveu Model" begin
    scheme = TRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-3
end

# === BTRG ===
@testset "BTRG - Gross-Neveu Model" begin
    scheme = BTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-4
end

# === HOTRG ===
@testset "HOTRG - Gross-Neveu Model" begin
    scheme = HOTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0; scalefactor = 4.0) ≈ f_bench rtol = 1.0e-3
end

# === ATRG ===
@testset "ATRG - Gross-Neveu Model" begin
    scheme = ATRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))
    @test free_energy(data, 1.0; scalefactor = 4.0) ≈ f_bench rtol = 1.0e-2
end

# === LoopTNR ===
@testset "LoopTNR - Gross-Neveu Model" begin
    scheme = LoopTNR(T)
    data = run!(scheme, truncdim(8), maxiter(10))
    @test free_energy(data, 1.0) ≈ f_bench rtol = 1.0e-3
end
