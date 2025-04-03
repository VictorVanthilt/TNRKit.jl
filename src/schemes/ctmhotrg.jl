mutable struct CTMHOTRG <: TNRScheme
    T::InfinitePartitionFunction
    E::CTMRGEnv
    finalize!::Function
    ctmalg::PEPSKit.CTMRGAlgorithm
    function CTMHOTRG(T::InfinitePartitionFunction, E::CTMRGEnv; finalize=finalize!,
                      ctmalg=PEPSKit.SequentialCTMRG(; maxiter=20, tol=1e-8))
        return new(T, E, finalize, ctmalg)
    end
end

function step!(scheme::CTMHOTRG, trunc::TensorKit.TruncationScheme)
    Tranf = Environment_tranfermatrix(scheme.E, scheme.T)
    Trans = Tranf / tr(Tranf)

    U, _, _, εₗ = tsvd(Trans; trunc=trunc)
    _, _, Uᵣ, εᵣ = tsvd(adjoint(Trans); trunc=trunc)

    if εₗ > εᵣ
        U = adjoint(Uᵣ)
    end

    @tensor A[-1 -2; -3 -4] := scheme.T[1][1 5; -3 3] * conj(U[1 2; -1]) *
                               U[3 4; -4] *
                               scheme.T[1][2 -2; 5 4]

    scheme.T = InfinitePartitionFunction(A)

    @tensor scheme.E.edges[4][-1 -2; -3] := scheme.E.edges[4][-1 1; 3] *
                                            scheme.E.edges[4][3 2; -3] * U[2 1; -2]
    @tensor scheme.E.edges[2][-1 -2; -3] := scheme.E.edges[2][-1 1; 3] *
                                            scheme.E.edges[2][3 2; -3] * conj(U[1 2; -2])

    scheme.E, = leading_boundary(scheme.E, scheme.T, scheme.ctmalg)

    return scheme
end

function Environment_tranfermatrix(env::CTMRGEnv, Z::InfinitePartitionFunction)
    C_NW = env.corners[1]
    C_NE = env.corners[2]
    C_SE = env.corners[3]
    C_SW = env.corners[4]
    E_N = env.edges[1]
    E_E = env.edges[2]
    E_S = env.edges[3]
    E_W = env.edges[4]

    T = Z[1]

    return PEPSKit.@autoopt @tensor Trans[D_WNE D_WSE; D_ENW D_ESW] := E_W[χ_WSW D_WSW;
                                                                           χ_WN] *
                                                                       E_W[χ_WN D_WNW;
                                                                           χ_WNW] *
                                                                       C_NW[χ_WNW; χ_NNW] *
                                                                       E_N[χ_NNW D_NNW;
                                                                           χ_NN] *
                                                                       E_N[χ_NN D_NNE;
                                                                           χ_NNE] *
                                                                       C_NE[χ_NNE; χ_ENE] *
                                                                       E_E[χ_ENE D_ENE;
                                                                           χ_EE] *
                                                                       E_E[χ_EE D_ESE;
                                                                           χ_ESE] *
                                                                       C_SE[χ_ESE; χ_SSE] *
                                                                       E_S[χ_SSE D_SSE;
                                                                           χ_SS] *
                                                                       E_S[χ_SS D_SSW;
                                                                           χ_SSW] *
                                                                       C_SW[χ_SSW; χ_WSW] *
                                                                       T[D_WNW D_NWSW;
                                                                         D_NNW D_ENW] *
                                                                       T[D_WSW D_SSW;
                                                                         D_NWSW D_ESW] *
                                                                       T[D_WNE D_NESE;
                                                                         D_NNE D_ENE] *
                                                                       T[D_WSE D_SSE;
                                                                         D_NESE D_ESE]
end
