@testset "FunZN in Rep[DN] Q-system property" begin
    N = 4
    FunZN, m = TNRKit.FunZN_Dihedral_even(N)
    @tensor AA_A[1; 2 3 4] := m[middle; 2 3] * m[1; middle 4]
    @tensor A_AA[1; 2 3 4] := m[middle; 3 4] * m[1; 2 middle]

    @test AA_A ≈ A_AA # Associativity is satisfied

    @test m * m' ≈ id(FunZN) # Isometry is satisfied

    N = 5
    FunZN, m = TNRKit.FunZN_Dihedral_odd(N)

    @tensor AA_A[1; 2 3 4] := m[middle; 2 3] * m[1; middle 4]
    @tensor A_AA[1; 2 3 4] := m[middle; 3 4] * m[1; 2 middle]

    @test AA_A ≈ A_AA # Associativity is satisfied

    @test m * m' ≈ id(FunZN) # Isometry is satisfied
end
