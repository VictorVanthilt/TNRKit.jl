using Test
using TNRKit
using TensorKit
using QuadGK

# include("spaces.jl") # do they give spacemismatches?
# include("schemes.jl") # do they give the correct results (with the expected accuracy)?
# include("schemes_triangular.jl") # do they give the correct results (with the expected accuracy)?
include("schemes_kagome.jl")
# include("models.jl") # do they give the correct results (with the expected accuracy)?
# include("fermions.jl") # do they give the correct results (with the expected accuracy)?
