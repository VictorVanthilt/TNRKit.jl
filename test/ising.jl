println("-------------")
println(" Ising Model ")
println("-------------")

criterion_f(steps::Int, data) = abs(log(data[end]) * 2.0^(1 - steps))

T = classical_ising_symmetric(Ising_βc)

# TRG
@testset "TRG - Ising Model" begin
    scheme = TRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 2e-6
end

# BTRG
@testset "BTRG - Ising Model" begin
    scheme = BTRG(T, -0.5)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-8
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    scheme = HOTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-7
end

# ATRG
@testset "ATRG - Ising Model" begin
    scheme = ATRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 3e-6
end

# GILTTNR
@testset "GILTTNR - Ising Model" begin
    scheme = GILTTNR(T; giltcrit=trivial_convcrit(1e-2) & maxiter(15))
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 2e-6
end

# SLoopTNR
@testset "SLoopTNR - Ising Model" begin
    scheme = SLoopTNR(classical_ising(Ising_βc)) # SLoopTNR is not compatible yet with symmetric tensors
    data = run!(scheme, truncdim(8), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 1e-5
end

@testset "LoopTNR - Ising Model" begin
    scheme = LoopTNR(T, T)
    data = run!(scheme, truncdim(8), maxiter(15); verbosity=2)

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 1e-5
end
