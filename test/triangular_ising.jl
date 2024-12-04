using Revise, TensorKit, CSV, Plots, DataFrames
includet("../src/TRGKit.jl")
using .TRGKit

trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))

exact_lnz = 0.3230659669

# stop when converged or after 50 steps, whichever comes first
stopping_criterion = convcrit(1e-20, trg_f)

χs = [8, 16, 24, 32]

lnz_trgs = []
lnz_btrgs = []
lnz_trgerrors = []
lnz_btrgerrors = []
for χ in χs
    scheme_TRG = TRG(triangle_bad())
    data_trg = run!(scheme_TRG, truncdim(χ), stopping_criterion; finalize_beginning=false)
    
    scheme_btrg = BTRG(triangle_bad(), -0.5)
    data_btrg = run!(scheme_btrg, truncdim(χ), stopping_criterion; finalize_beginning=false)

    lnz_trg = 0
    for (i, d) in enumerate(data_trg)
        lnz_trg+= log(d) * 2.0^(-i)        
    end
    @show lnz_trg 

    lnz_btrg = 0
    for (i, d) in enumerate(data_btrg)
        lnz_btrg+= log(d) * 2.0^(-i)        
    end
    @show lnz_btrg 

    push!(lnz_trgs, lnz_trg)
    push!(lnz_btrgs, lnz_btrg)
    push!(lnz_trgerrors, abs(lnz_trg - exact_lnz))
    push!(lnz_btrgerrors, abs(lnz_btrg - exact_lnz))

end

CSV.write("data/triangle_atsushi.csv", DataFrame(χ=χs, lnz_trg=lnz_trgs, lnz_btrg=lnz_btrgs, lnz_trgerror = lnz_trgerrors, lnz_btrgerror = lnz_btrgerrors))

data = CSV.read("data/triangle_atsushi.csv", DataFrame)
p1 = scatter(data.χ, data.lnz_trg, xlabel = "χ", ylabel = "Free energy", label = "TRG")
p2 = scatter!(data.χ, data.lnz_btrg, xlabel = "χ", ylabel = "Free energy", label = "BTRG")
savefig("atsushi.png") # save the fig referenced by plot_ref as filename_string (such as "output.png")

p3 = scatter(data.χ, data.lnz_trgerror, xlabel = "χ", ylabel = "Error", label = "TRG")
p4 = scatter!(data.χ, data.lnz_btrgerror, xlabel = "χ", ylabel = "Error", label = "BTRG")
plot!(p3, yscale=:log10)
plot!(p4, yscale=:log10)
savefig("atsushi_error.png") # save the fig referenced by plot_ref as filename_string (such as "output.png")