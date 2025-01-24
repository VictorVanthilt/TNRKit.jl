println("--------")
println(" spaces ")
println("--------")

# stuff different kinds of spaces into the schemes (and do 100 steps)
A = classical_ising(Ising_βc)
B = classical_ising_symmetric(Ising_βc)
C = gross_neveu_start(0, 1, 0)

models = [A, B, C]
schemes = [TRG, BTRG, HOTRG, ATRG, GILTTNR]

# The tests below check that the schemes don't lead to spacemismatches
for S in schemes
    @testset "$(S) - spaces" begin
        for T in models
            @eval begin
                @test isa(run!($S($(T)), truncdim(7), maxiter(100)), Any)
            end
        end
    end
end
