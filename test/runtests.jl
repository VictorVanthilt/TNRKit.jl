using Test
using TRGKit
using TensorKit

using QuadGK

include("spaces.jl") # do they give spacemismatches?
include("ising.jl") # do they give the correct results (with the expected accuracy)?
