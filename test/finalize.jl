println("---------------------")
println(" two by two finalize ")
println("---------------------")

criterion_f(steps::Int, data) = abs(log(data[end]) * 2.0^(1 - steps))

T = classical_ising_symmetric()

# TRG
@testset "TRG - Ising Model" begin
    scheme = TRG(T; finalize = (finalize_two_by_two!))
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 2.0e-6
end

# BTRG
@testset "BTRG - Ising Model" begin
    scheme = BTRG(T, -0.5; finalize = (finalize_two_by_two!))
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6.0e-8
end

# HOTRG
@testset "HOTRG - Ising Model" begin
    scheme = HOTRG(T; finalize = (finalize_two_by_two!))
    data = run!(scheme, truncdim(16), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 4.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 6.0e-7
end

# ATRG
@testset "ATRG - Ising Model" begin
    scheme = ATRG(T; finalize = (finalize_two_by_two!))
    data = run!(scheme, truncdim(24), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 4.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 3.0e-6
end

# ImpurityHOTRG
@testset "ImpurityHOTRG - Ising Model" begin
    T = ising_magnetisation(ising_βc; impurity = false)
    T_imp1 = ising_magnetisation(ising_βc; impurity = true)
    scheme = ImpurityHOTRG(T, T_imp1, T_imp1, T)
    data = run!(scheme, truncdim(16), maxiter(25))

    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d[1]) * 4.0^(1 - i)
    end

    fs = lnz * -1 / ising_βc

    relerror = abs((fs - f_onsager) / f_onsager)
    @test relerror < 3.0e-6
end

@testset "Impurity HOTRG - Magnetisation" begin
    beta1 = 0.2
    T = ising_magnetisation(beta1; impurity = false)
    T_imp_order1_1 = ising_magnetisation(beta1; impurity = true)
    T_imp_order2 = ising_magnetisation(beta1; impurity = false)
    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)
    data = run!(scheme, truncdim(8), maxiter(25))

    m2_highT = data[end][4] / data[end][1]
    m_actual = 0.0
    relerror = abs((m2_highT - m_actual) / m_actual)
    @test relerror < 1.0e-2


    beta2 = 1.0
    T = ising_magnetisation(beta2; impurity = false)
    T_imp_order1_1 = ising_magnetisation(beta2; impurity = true)
    T_imp_order2 = ising_magnetisation(beta2; impurity = false)
    scheme = ImpurityHOTRG(T, T_imp_order1_1, T_imp_order1_1, T_imp_order2)
    data = run!(scheme, truncdim(8), maxiter(25))
    m2_lowT = data[end][4] / data[end][1]
    m_actual = 1.0
    relerror = abs((m2_lowT - m_actual) / m_actual)
    @test relerror < 1.0e-2
end
