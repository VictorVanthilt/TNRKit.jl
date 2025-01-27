using Test
using TNRKit
using TensorKit

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end

@time begin
    if GROUP == "ALL" || GROUP == "SPACES"
        @time include("spaces.jl") # do they give spacemismatches?
    end
    if GROUP == "ALL" || GROUP == "ISING"
        @time include("ising.jl") # do they give the correct results (with the expected accuracy)?
    end
    if GROUP == "ALL" || GROUP == "FINALIZE"
        @time include("finalize.jl") # do they give the correct results (with the expected accuracy)?
    end
end
