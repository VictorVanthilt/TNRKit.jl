println("--------")
println(" spaces ")
println("--------")

# stuff different kinds of spaces into the schemes (and do 100 steps)
A = ("Normal Ising", classical_ising(Ising_βc))
B = ("Ising Symmetric", classical_ising_symmetric(Ising_βc))
C = ("Gross Neveu", gross_neveu_start(0, 1, 0))
D = ("Sixvertex U1", sixvertex(ComplexF64, U1Irrep))
E = ("Sixvertex CU1", sixvertex(ComplexF64, CU1Irrep))

models = [A, B, C, D, E]
schemes = [TRG, BTRG, HOTRG, ATRG, GILTTNR]

# The tests below check that the schemes don't lead to spacemismatches
for S in schemes
    @testset "$(S) - spaces" begin
        for T in models
            @eval begin
                println("--- $($(T[1])) ---")
                @test isa(run!($S($(T[2])), truncdim(7), maxiter(25)), Any)
            end
        end
    end
end
