println("-----------------------")
println(" Testing all entropies ")
println("-----------------------")

@testset "von Neumann entropy of a TensorMap" begin
    M = TensorMap([0.5 0 0; 0 0.5 0; 0 0 0.5], ℂ^3 ← ℂ^3)
    S_von, spec = VN_entropy(M)
    @test isapprox(S_von, log(3))
    @test isapprox(spec.data, [1.0, 1.0, 1.0])

    N = TensorMap([10.0 0.0 0; 0.0 0.0 0; 0 0 0], ℂ^3 ← ℂ^3)
    S_von_N, spec_N = VN_entropy(N)
    @test isapprox(S_von_N, 0.0)
    @test isapprox(spec_N.data, [0.0, 0.0, 1.0])
end

@testset "loop entropy" begin
    T = classical_ising()
    scheme = LoopTNR(T)
    data = run!(scheme, truncrank(8), maxiter(8), LoopParameters(); verbosity=1)
    entropies_circ, specs_circ, entropies_rad, specs_rad = loop_entropy(scheme)

    @test true # Just check that the function runs without errors
end