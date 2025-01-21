println("---------------------")
println(" two by two finalize ")
println("---------------------")

# Onsager solution
function onsager_integrand(θ, T)
    k = 1 / sinh(2 / T)^(2)
    integrand = 1 / (2π) * log(cosh(2 / T)^2 + 1 / k * sqrt(1 + k^2 - 2k * cos(2 * θ)))
    return integrand
end

function onsager_free_energy(β)
    return -(quadgk(θ -> onsager_integrand(θ, 1 / β), 0, π)[1] + log(2) / 2) / β
end

criterion_f(steps::Int, data) = abs(log(data[end]) * 2.0^(1 - steps))

T = classical_ising_symmetric(Ising_βc)
fs_onsager = onsager_free_energy(Ising_βc)

# TRG
@testset "TRG - Ising Model" begin
    scheme = TRG(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - fs_onsager) / fs_onsager)
    @test relerror < 2e-6
end

# BTRG
@testset "BTRG - Ising Model" begin
    scheme = BTRG(T, -0.5; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - fs_onsager) / fs_onsager)
    @test relerror < 6e-8
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    scheme = HOTRG(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(16), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - fs_onsager) / fs_onsager)
    @test relerror < 6e-7
end

# ATRG
@testset "ATRG - Ising Model" begin
    scheme = ATRG(T; finalize=finalize_two_by_two!)
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / Ising_βc

    relerror = abs((fs - fs_onsager) / fs_onsager)
    @test relerror < 3e-6
end
