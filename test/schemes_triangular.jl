# c6vCTM_triangular
@testset "c6vCTM_triangular - Ising Model" begin
    for method in [classical_ising_triangular classical_ising_triangular_symmetric]
        for projectors in [:twothirds :full]
            for conditioning in [true false]
                T_flipped = method(ising_βc_triangular)

                scheme = c6vCTM_triangular(T_flipped)
                lz = run!(scheme, truncdim(20), maxiter(100); projectors, conditioning)

                fs = lz * -1 / ising_βc_triangular
                @test fs ≈ f_onsager_triangular rtol = 1.0e-4
            end
        end
    end
end

# CTM_triangular
@testset "CTM_triangular - Ising Model" begin
    for method in [classical_ising_triangular classical_ising_triangular_symmetric]
        for projectors in [:twothirds :full]
            for conditioning in [true false]
                T_flipped = method(ising_βc_triangular)

                scheme = CTM_triangular(T_flipped)
                lz = run!(scheme, truncdim(20), maxiter(100); projectors, conditioning)

                fs = lz * -1 / ising_βc_triangular
                @test fs ≈ f_onsager_triangular rtol = 1.0e-4
            end
        end
    end
end
