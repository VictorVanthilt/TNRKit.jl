println("-------------")
println(" Ising Model ")
println("-------------")

criterion_f(steps::Int, data) = abs(log(data[end]) * 2.0^(1 - steps))

T = classical_ising_symmetric()

function free_energy(data, β)
    lnz = 0
    for (i, z) in enumerate(data)
        lnz += log(z) * 2.0^(1 - i)
    end
    return -lnz / β
end
# TRG
@testset "TRG - Ising Model" begin
    scheme = TRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

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

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-8
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    scheme = HOTRG(T)
    data = run!(scheme, truncdim(16), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 4.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-7
end

# ATRG
@testset "ATRG - Ising Model" begin
    scheme = ATRG(T)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 4.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 3e-6
end

# c4CTM
@testset "c4CTM - Ising Model" begin
    scheme = c4CTM(T)
    lz = run!(scheme, truncdim(24), trivial_convcrit(1e-9); verbosity=1)

    fs = lz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-8
end

# rCTM
@testset "rCTM - Ising Model" begin
    scheme = rCTM(T)
    lz = run!(scheme, truncdim(24), trivial_convcrit(1e-9); verbosity=1)

    fs = lz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6e-8
end

#LoopTNR
@testset "LoopTNR - Ising Model" begin
    scheme = LoopTNR(T)

    entanglement_function(steps, data) = abs(data[end])
    entanglement_criterion = maxiter(100) & convcrit(1e-15, entanglement_function)
    loop_criterion = maxiter(5) & convcrit(1e-10, entanglement_function)

    data = run!(scheme, truncdim(8), truncbelow(1e-12), maxiter(25), entanglement_criterion,
                loop_criterion)

    fs = free_energy(data, ising_βc)

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 1e-4
end

@testset "SLoopTNR - Ising Model" begin
    T = classical_ising_inv()
    scheme = SLoopTNR(T)

    data = run!(scheme, truncdim(4), maxiter(25))

    fs = free_energy(data, ising_βc)

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 1e-5
end

# CTMHOTRG
@testset "CTMHOTRG - Ising Model" begin
    Z = InfinitePartitionFunction(T)
    χ = 16
    χenv = Z2Space(0 => χ / 2, 1 => χ / 2)
    env0 = CTMRGEnv(Z, χenv)
    ctmalg = SequentialCTMRG(; maxiter=10000, tol=1e-8, verbosity=3)
    env, = leading_boundary(env0, Z, ctmalg)
    scheme = CTMHOTRG(Z, env;
                      ctmalg=SequentialCTMRG(; maxiter=50, tol=1e-8))
    data = run!(scheme, truncdim(χ), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 1e-6
end

# ATRG_3D
@testset "ATRG_3D - Ising Model" begin
    T_3D = classical_ising_symmetric_3D()
    scheme = ATRG_3D(T_3D)
    data = run!(scheme, truncdim(12), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 8.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc_3D
    @show fs
    f_benchmark = -3.515
    relerror = abs((fs - f_benchmark) / f_benchmark)
    @test relerror < 1e-3
end
