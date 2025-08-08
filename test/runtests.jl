using Test
using TNRKit
using TensorKit
using PEPSKit: InfinitePartitionFunction, CTMRGEnv, SequentialCTMRG, leading_boundary
using QuadGK

include("spaces.jl") # do they give spacemismatches?
include("schemes.jl") # do they give the correct results (with the expected accuracy)?
include("models.jl") # do they give the correct results (with the expected accuracy)?
include("fermions.jl") # do they give the correct results (with the expected accuracy)?
