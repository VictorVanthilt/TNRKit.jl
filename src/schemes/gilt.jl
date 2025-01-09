# Graph Independent Local Truncation Hauru Delcamp

mutable struct GILT <: TRGScheme
    T1::TensorMap
    T2::TensorMap
    T3::TensorMap
    T4::TensorMap

    ε::Float64
    function GILT(T::TensorMap; ε=5e-8)
        new(copy(T), copy(T), copy(T), copy(T), ε)
    end
end

function step!(scheme::GILT, trunc::TensorKit.TruncationScheme)
    # Environment: top leg broken
    @tensor E[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10] := scheme.T1[-2 1; -10 -1] *
                                                  scheme.T2[-9 3; -7 -8] *
                                                  scheme.T3[2 -5; -6 3] *
                                                  scheme.T4[-3 -4; 2 1]
    S², U = eigh(adjoint(E) * E)

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[-1] * (S² * inv(epsid + S²))[-1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]

    U, S, V, _ = tsvd(R′)
    sqrtS = sqrt(S)

    @tensor scheme.T1[-1 -2; -3 -4] := scheme.T1[-1 -2; 1 -4] * U[1; 2] * sqrtS[2; -3]
    @tensor scheme.T2[-1 -2; -3 -4] := sqrtS[-1; 1] * V[1; 2] * scheme.T2[2 -2; -3 -4]

    # Environment: right leg broken
    @tensor E[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10] := scheme.T1[-4 2; 1 -3] *
                                                  scheme.T2[1 -10; -1 -2] *
                                                  scheme.T3[3 -7; -8 -9] *
                                                  scheme.T4[-5 -6; 3 2]

    S², U = eigh(adjoint(E) * E)

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[-1] * (S² * inv(epsid + S²))[-1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]
    R′ = transpose(R′)
    U, S, V, _ = tsvd(R′)
    sqrtS = sqrt(S)

    @tensor scheme.T2[-1 -2; -3 -4] := scheme.T2[-1 1; -3 -4] * U[2; 1] * sqrtS[-2; 2]
    @tensor scheme.T3[-1 -2; -3 -4] := sqrtS[2; -4] * V[1; 2] * scheme.T3[-1 -2; -3 1]

    # Environment: bottom leg broken
    @tensor E[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10] := scheme.T1[-6 3; 2 -5] *
                                                  scheme.T2[2 1; -3 -4] *
                                                  scheme.T3[-10 -1; -2 1] *
                                                  scheme.T4[-7 -8; -9 3]

    S², U = eigh(adjoint(E) * E)

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[-1] * (S² * inv(epsid + S²))[-1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]
    R′ = transpose(R′)
    U, S, V, _ = tsvd(R′)
    sqrtS = sqrt(S)

    @tensor scheme.T3[-1 -2; -3 -4] := sqrtS[-1; 1] * U[1; 2] * scheme.T3[2 -2; -3 -4]
    @tensor scheme.T4[-1 -2; -3 -4] := V[1; 2] * sqrtS[2; -3] * scheme.T4[-1 -2; 1 -4]

    # Environment: left leg broken
    @tensor E[-1 -2 -3 -4 -5 -6 -7 -8; -9 -10] := scheme.T1[-8 -9; 3 -7] *
                                                  scheme.T2[3 2; -5 -6] *
                                                  scheme.T3[1 -3; -4 2] *
                                                  scheme.T4[-1 -2; 1 -10]

    S², U = eigh(adjoint(E) * E)

    @plansor t[-1] := U[1 1; -1]

    epsid = scheme.ε^2 * id(domain(S²))

    @tensor t′[-1] := t[-1] * (S² * inv(epsid + S²))[-1; -1]

    @plansor R′[-1; -2] := adjoint(U)[1; -1 -2] * t′[1]
    U, S, V, _ = tsvd(R′)
    sqrtS = sqrt(S)

    @tensor scheme.T4[-1 -2; -3 -4] := scheme.T4[-1 -2; -3 1] * U[1; 2] * sqrtS[2; -4]
    @tensor scheme.T1[-1 -2; -3 -4] := sqrtS[-2; 1] * V[1; 2] * scheme.T1[-1 2; -3 -4]

    return scheme
end