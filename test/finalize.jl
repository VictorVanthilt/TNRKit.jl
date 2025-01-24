println("---------------------")
println(" two by two finalize ")
println("---------------------")

criterion_f(steps::Int, data) = abs(log(data[end]) * 2.0^(1 - steps))

T = classical_ising_symmetric(Ising_βc)

# TRG
@testset "TRG - Ising Model" begin
    scheme = TRG(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-(i + 1))
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 2e-6
end

# BTRG
@testset "BTRG - Ising Model" begin
    scheme = BTRG(T, -0.5; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-(i + 1))
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-8
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    scheme = HOTRG(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(16), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-(i + 1))
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-7
end

# ATRG
@testset "ATRG - Ising Model" begin
    scheme = ATRG(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-(i + 1))
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 3e-6
end

# GILTTNR
@testset "GILTTNR - Ising Model" begin
    scheme = GILTTNR(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25); verbosity=2)

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-(i + 1))
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 2e-6
end
