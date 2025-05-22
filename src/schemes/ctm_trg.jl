using TNRKit, TensorKit
include("oblique_projector.jl")

mutable struct ctm_TRG{A,S} <: TNRScheme
    T::TensorMap{A,S,2,2}
    C2::TensorMap{A,S,1,1}
    E1::TensorMap{A,S,2,1}
    E2::TensorMap{A,S,2,1}
    χenv::Int64
    function ctm_TRG(T::TensorMap{A,S,2,2},
                     χenv::Int64;
                     ctm_iter=2e4,
                     ctm_tol=1e-9,) where {A,S}
        if eltype(T) != Float64
            @error "This scheme only support tensors with real numbers"
        end
        scheme_init = TNRKit.rCTM(T)
        @info "Finding the environment using rCTM..."
        TNRKit.run!(scheme_init,
                    truncdim(χenv),
                    trivial_convcrit(ctm_tol) & maxiter(ctm_iter);
                    verbosity=0,)
        @info "rCTM finished"
        C2, E1, E2 = scheme_init.C2, scheme_init.E1, scheme_init.E2
        @assert BraidingStyle(sectortype(T)) == Bosonic() "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for rCTM"
        return new{A,S}(T, C2, E1, E2, χenv)
    end
end

function tr_tensor(T; inv=false)
    if inv
        @tensoropt tr4 = T[1 2; 3 4] * conj(T[5 2; 3 6]) * conj(T[1 7; 8 4]) * T[5 7; 8 6]
        return (abs(tr4))^(1 / 4)
    else
        return @tensor T[1 2; 2 1]
    end
end

function corner_matrix(scheme::ctm_TRG)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.E1[-1 3; 1] * scheme.C2[1; 2] *
                                            scheme.E2[2 4; -3] * scheme.T[-2 -4; 3 4]
    return mat
end

function find_UVt(scheme::ctm_TRG, trunc)
    mat = corner_matrix(scheme)
    U, S, Vt = tsvd(mat; trunc=trunc & truncbelow(1e-20))
    return mat, U, S, Vt
end

function rctm_step!(scheme; truc=truncdim(dim(scheme.C2.space.domain)))
    mat, U, S, Vt = find_UVt(scheme, truncdim(16))
    scheme.C2 = adjoint(U) * mat * adjoint(Vt)
    @tensor opt = true scheme.E1[-1 -2; -3] := scheme.E1[1 5; 3] * scheme.T[2 -2; 5 4] *
                                               U[3 4; -3] * conj(U[1 2; -1])
    @tensor opt = true scheme.E2[-1 -2; -3] := scheme.E2[1 5; 3] * scheme.T[-2 4; 2 5] *
                                               conj(Vt[-3; 3 4]) * Vt[-1; 1 2]
    scheme.C2 /= norm(scheme.C2)
    scheme.E1 /= norm(scheme.E1)
    scheme.E2 /= norm(scheme.E2)
    return S
end

function Levin_decomposition(T, trunc;)
    U, S, Vt = tsvd(T, (1, 3), (2, 4); trunc=trunc)

    S1 = U * sqrt(S)
    S2 = sqrt(S) * Vt
    return S1, S2
end

function insert_PtoS(scheme, trunc; enlarge=true)
    S1, S2 = Levin_decomposition(scheme.T, notrunc())

    @tensoropt env_left_top[-1 -2; -3] := S1[3 4; -3] *
                                          scheme.E1[2 4; -1] *
                                          conj(scheme.E2[1 3; 5]) *
                                          conj(scheme.C2[2; 1]) *
                                          scheme.C2[-2; 5]
    @tensoropt env_right_bottom[-1; -2 -3] := S2[-1; 1 2] *
                                              scheme.E2[3 2; 4] *
                                              conj(scheme.E1[-3 1; 5]) *
                                              scheme.C2[-2; 3] *
                                              conj(scheme.C2[5; 4])
    if enlarge
        P1, P2 = find_P1P2(env_left_top, env_right_bottom, (3,), (1,),
                           truncdim(trunc.dim * 2))
    else
        P1, P2 = find_P1P2(env_left_top, env_right_bottom, (3,), (1,), trunc)
    end
    return S1 * P1, P2 * S2
end

# find the projector for bundling two bonds in the vertical and horizontal directions
# I checked the modified version provides the same accuracy for free energy. (and cheaper)
# I persue this becuase this is compatible with entanglement filtering
function hotrg_projector(scheme, trunc; modified=true)
    if modified
        return hotrg_projector_modified(scheme, trunc)
    else
        mat = corner_matrix(scheme)

        matv2 = mat * adjoint(mat)
        Pv1, Pv2 = find_P1P2(matv2, adjoint(matv2), (2, 4), (4, 2), trunc)
        math2 = adjoint(mat) * mat
        Ph1, Ph2 = find_P1P2(math2, adjoint(math2), (2, 4), (4, 2), trunc)
        return Pv1, Pv2, Ph1, Ph2
    end
end

function hotrg_projector_modified(scheme, trunc)
    @tensor mat[-1; -2 -3] := scheme.E1[-1; -2 1]*scheme.C2[1; -3]
    math2 = adjoint(mat) * mat
    Ph1, Ph2 = find_P1P2(math2, adjoint(math2), (1, 3), (3, 1), trunc)

    @tensor mat[-1 -2; -3] := scheme.C2[-1; 1] * scheme.E2[1 -2; -3]
    matv2 = mat * adjoint(mat)
    Pv1, Pv2 = find_P1P2(matv2, adjoint(matv2), (2, 4), (4, 2), trunc)
    return Pv1, Pv2, Ph1, Ph2
end

function step!(scheme::ctm_TRG,
               trunc;
               sweep=30,
               χenv=dim(scheme.C2.space.domain),
               enlarge=true,
               inv=false,
               modified=true,)
    S1, S2 = insert_PtoS(scheme, trunc; enlarge=enlarge)
    Pv1, Pv2, Ph1, Ph2 = hotrg_projector(scheme, trunc; modified)
    # My apologies for the unreadable contraction
    # It just conbines tensors and projectors to build the new tensor
    @tensoropt Tnew[-1 -2; -3 -4] := Pv1[2 3; -1] *
                                     S2[13; 1 3] *
                                     conj(S2[16; 1 2]) *
                                     Pv2[-4; 11 12] *
                                     S2[15; 10 11] *
                                     conj(S2[14 10 12]) *
                                     Ph1[5 6; -2] *
                                     S1[4 5; 13] *
                                     conj(S1[4 6; 14]) *
                                     Ph2[-3; 8 9] *
                                     S1[7 9; 15] *
                                     conj(S1[7 8; 16])
    @tensoropt E1new[-1 -2; -3] := Ph1[1 2; -2] * scheme.E1[3 2; -3] *
                                   conj(scheme.E1[3 1; -1])
    @tensoropt E2new[-1 -2; -3] := Pv1[1 2; -2] * scheme.E2[-1 1; 3] *
                                   conj(scheme.E2[-3 2; 3])
    scheme.T = Tnew
    scheme.E1 = E1new
    scheme.E2 = E2new

    tr_norm = tr_tensor(scheme.T; inv=inv)
    scheme.T /= tr_norm
    scheme.E1 /= norm(scheme.E1)
    scheme.E2 /= norm(scheme.E2)
    for _ in 0:sweep
        rctm_step!(scheme)
    end
    return tr_norm
end

function run!(scheme::ctm_TRG,
              trunc,
              criterion;
              sweep=30,
              enlarge=true,
              return_cft=false,
              inv=false,
              conv_criteria=1e-12,
              modified=true,)
    area = 1
    lnz = 0.0
    cft = []

    for i in 1:criterion.n
        area *= 4.0
        tr_norm = step!(scheme, trunc; sweep=sweep, enlarge=enlarge, inv=inv, modified)
        lnz += log(tr_norm) / area
        if return_cft
            push!(cft, cft_data(scheme; unitcell=2))
        end
        if abs(log(abs(tr_norm)) / area) <= conv_criteria
            @info "CTM-TRG converged after $i iterations!"
            break
        end
    end
    if return_cft
        return lnz, cft
    else
        return lnz
    end
end
