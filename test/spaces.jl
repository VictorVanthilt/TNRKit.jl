println("--------")
println(" spaces ")
println("--------")

# stuff different kinds of spaces into the schemes (and do 100 steps)
A = classical_ising(Ising_βc)
B = classical_ising_symmetric(Ising_βc)
C = gross_neveu_start(0, 1, 0)

models = [A, B, C]

# The tests below check that the schemes don't lead to spacemismatches
@testset "TRG - spaces" begin
    for T in models
        @eval begin
            @test isa(run!(TRG($(T)), truncdim(7), maxiter(100)), Any)
        end
    end
end

@testset "BTRG - spaces" begin
    for T in models
        @eval begin
            @test isa(run!(BTRG($(T)), truncdim(7), maxiter(100)), Any)
        end
    end
end

@testset "HOTRG - spaces" begin
    for T in models
        @eval begin
            @test isa(run!(HOTRG($(T)), truncdim(7), maxiter(100)), Any)
        end
    end
end
