# c6CTM
@testset "c6CTM - Ising Model" begin
    T = classical_ising_symmetric_triangular()
    T = flip(T, (1,2,3))
    @info "c6CTM ising free energy"
    scheme = c6CTM_triangular(T)
    lz = run!(scheme, truncdim(24), trivial_convcrit(1.0e-9) & maxiter(1000); verbosity = 1)

    fs = lz * -1 / ising_βc
end
