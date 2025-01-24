using TensorKit, TRGKit

T = randn(ComplexF64, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)
T = classical_ising_symmetric(Ising_βc)
T = gross_neveu_start(0, 1, 0)

scheme = GILTTNR(T; ε=5e-8);

TRGKit.step!(scheme, truncbelow(5e-8));

scheme.T1
scheme.T3
scheme.T1 ≈ scheme.T3

for _ in 1:100
    TRGKit.step!(scheme, truncbelow(scheme.ε))
end

@tensor E[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10] := scheme.T1[-2 1; -10 -1] *
                                              scheme.T2[-9 3; -7 -8] *
                                              scheme.T3[2 -5; -6 3] *
                                              scheme.T4[-3 -4; 2 1];
S², U = eigh(adjoint(E) * E)

space(S²)
space(U)

@plansor t[-1] := U[1 1; -1]

epsid = scheme.ε^2 * id(domain(S²))

@tensor t′[-1] := t[1] * (S² * inv(epsid + S²))[1; -1]

@plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]

U, S, V, _ = tsvd(R′; trunc=truncbelow(scheme.ε))
sqrtS = sqrt(S)

@tensor scheme.T1[-1 -2; -3 -4] := scheme.T1[-1 -2; 1 -4] * U[1; 2] * sqrtS[2; -3]
@tensor scheme.T2[-1 -2; -3 -4] := sqrtS[-1; 1] * V[1; 2] * scheme.T2[2 -2; -3 -4]

scheme.T1
scheme.T2
scheme.T3
scheme.T4
