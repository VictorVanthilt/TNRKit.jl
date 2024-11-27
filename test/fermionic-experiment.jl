using Revise, TensorKit, DataFrames, CSV
includet("../src/TRGKit.jl")
using .TRGKit

function fermionic_trg_finalize!(scheme::TRGScheme)
    n = norm(@tensor scheme.T[1 2; 1 2])
    scheme.T /= n
    return n
end

function fermionic_btrg_finalize!(scheme::TRGScheme)
    scheme.T = permute(scheme.T, (1, 2), (3, 4))

    n = norm(@tensor scheme.T[1 2; 3 4] * scheme.S1[4; 2] * scheme.S2[3; 1])
    scheme.T /= n
    return n
end

custom_convcrit(steps::Int, data) = abs(log(data[end]) * 2.0^(1-steps))

ms = [0, -1]
χs = [20, 30, 40, 50, 60, 70]

for m in ms
    lnz_trgs = []
    lnz_btrgs = []
    for χ in χs
        T = gross_neveu_start(0, m, 0)
        trg = TRG(copy(T); f=fermionic_trg_finalize!)
        btrg = BTRG(copy(T), -0.5;  f=fermionic_btrg_finalize!)

        data_trg = run!(trg, truncdim(χ),  stop=convcrit(1e-20, custom_convcrit))
        data_btrg = run!(btrg, truncdim(χ), stop=convcrit(1e-20, custom_convcrit))

        lnz_trg = 0
        for (i, d) in enumerate(data_trg)
            lnz_trg += log(d) * 2.0^(1-i)
        end

        lnz_btrg = 0
        for (i, d) in enumerate(data_btrg)
            lnz_btrg += log(d) * 2.0^(1-i)
        end
        push!(lnz_trgs, lnz_trg)
        push!(lnz_btrgs, lnz_btrg)
    end
    CSV.write("data/fermionic_m$(m).csv", DataFrame(χ=χs, lnz_trg=lnz_trgs, lnz_btrg=lnz_btrgs))
end