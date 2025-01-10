using Revise, TensorKit, Plots, QuadGK, DataFrames, CSV
includet("../src/TRGKit.jl")
using .TRGKit

Tc = 2 / log(1 + sqrt(2))
Ts = collect(range(Tc - 0.2, Tc + 0.2, 100))
fs_btrg = zeros(length(Ts))
fs_trg = zeros(length(Ts))

χ = 24

for (i, T) in enumerate(Ts)
    scheme = BTRG(classical_ising(1 / T), -0.5;
                  stop=maxiter(100) & convcrit(1e-20, btrg_convcrit))
    data = run!(scheme, truncdim(χ))
    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-i)
    end
    fs_btrg[i] = lnz * -T
end

for (i, T) in enumerate(Ts)
    scheme = TRG(classical_ising(1 / T); stop=maxiter(100) & convcrit(1e-20, trg_convcrit))
    data = run!(scheme, truncdim(χ))
    lnz = 0
    for (i, d) in enumerate(data)
        lnz += log(d) * 2.0^(-i)
    end
    fs_trg[i] = lnz * -T
end

function onsager_integrand(θ, T)
    k = 1 / sinh(2 / T)^(2)
    integrand = 1 / (2π) * log(cosh(2 / T)^2 + 1 / k * sqrt(1 + k^2 - 2k * cos(2 * θ)))
    return integrand
end

# integrates onsagers exact solution using quadgk
function onsager_free_energy(Tarray)
    farray = zeros(length(Tarray))

    for i in eachindex(Tarray)
        farray[i] = quadgk(θ -> onsager_integrand(θ, Tarray[i]), 0, π)[1]
    end

    farray .+= log(2) / 2

    return -farray .* Tarray
end

CSV.write("data/relerror.csv",
          DataFrame(; Ts=Ts, fs_btrg=fs_btrg, fs_trg=fs_trg,
                    fs_onsager=onsager_free_energy(Ts)))
