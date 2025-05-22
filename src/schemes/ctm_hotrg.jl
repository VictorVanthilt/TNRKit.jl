using TNRKit, TensorKit
include("oblique_projector.jl")

mutable struct ctm_HOTRG{A,S} <: TNRScheme
    T::TensorMap{A,S,2,2}
    C2::TensorMap{A,S,1,1}
    E1::TensorMap{A,S,2,1}
    E2::TensorMap{A,S,2,1}
    χenv::Int64
    function ctm_HOTRG(T::TensorMap{A,S,2,2},
                       χenv::Int64;
                       ctm_iter=2e4,
                       ctm_tol=1e-9,
                       ctm_obc=false,
                       χenv_ini=2,) where {A,S}
        if eltype(T) != Float64
            @error "This scheme only supports tensors with real numbers"
        end
        scheme_init = TNRKit.rCTM(T)
        # if ctm_obc
        #     C, E1, E2 = rCTM_init_OBC(T;χenv_ini = χenv_ini)
        #     scheme_init.C2, scheme_init.E1, scheme_init.E2 = C, E1, E2
        # end

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

function abs_tensor!(T)
    T.data .= abs.(T.data)
    return T
end

function rCTM_init_OBC(T; χenv_ini=2)
    elt = typeof(T.data[1])
    Vp1 = space(T)[3]'
    Vp2 = space(T)[4]'
    S1 = sectortype(Vp1)
    V1 = Vect[S1](one(S1) => χenv_ini)
    S2 = sectortype(Vp2)
    V2 = Vect[S2](one(S2) => χenv_ini)

    C = TensorMap(randn, elt, V1 ← V2)
    E1 = TensorMap(randn, elt, V1 ⊗ Vp1 ← V1)
    E2 = TensorMap(randn, elt, V2 ⊗ Vp2 ← V2)

    return abs_tensor!(C), abs_tensor!(E1), abs_tensor!(E2)
end

function tr_tensor(T; inv=false)
    if inv
        @tensoropt tr4 = T[1 2; 3 4] * conj(T[5 2; 3 6]) * conj(T[1 7; 8 4]) * T[5 7; 8 6]
        return (abs(tr4))^(1 / 4)
    else
        return @tensor T[1 2; 2 1]
    end
end

function corner_matrix(scheme::ctm_HOTRG)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.E1[-1 3; 1] * scheme.C2[1; 2] *
                                            scheme.E2[2 4; -3] * scheme.T[-2 -4; 3 4]
    return mat
end

function find_UVt(scheme::ctm_HOTRG, trunc)
    mat = corner_matrix(scheme)
    U, S, Vt = tsvd(mat; trunc=trunc & truncbelow(1e-20))
    return mat, U, S, Vt
end

function vertical_move!(scheme, trunc)
    mat = corner_matrix(scheme)
    mat2 = mat * adjoint(mat)
    P1, P2 = find_P1P2(mat2, adjoint(mat2), (2, 4), (4, 2), trunc)
    @tensoropt Tnew[-1 -2; -3 -4] := P1[1 3; -1] * P2[-4; 2 4] * scheme.T[1 5; -3 2] *
                                     conj(scheme.T[3 5; -2 4])
    @tensoropt E2new[-1 -2; -3] := P1[1 2; -2] * scheme.E2[-1 1; 3] *
                                   conj(scheme.E2[-3 2; 3])

    scheme.T = Tnew
    return scheme.E2 = E2new
end

function horizontal_move!(scheme, trunc)
    mat = corner_matrix(scheme)
    mat2 = adjoint(mat) * mat
    P1, P2 = find_P1P2(mat2, adjoint(mat2), (2, 4), (4, 2), trunc)
    @tensoropt Tnew[-1 -2; -3 -4] := P1[3 4; -2] * P2[-3; 1 2] * scheme.T[5 4; 2 -4] *
                                     conj(scheme.T[5 3; 1 -1])
    @tensoropt E1new[-1 -2; -3] := P1[1 2; -2] * scheme.E1[3 2; -3] *
                                   conj(scheme.E1[3 1; -1])

    scheme.T = Tnew
    return scheme.E1 = E1new
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

function step!(scheme::ctm_HOTRG,
               trunc;
               sweep=30,
               χenv=dim(scheme.C2.space.domain),
               inv=false,)
    vertical_move!(scheme, trunc)
    horizontal_move!(scheme, trunc)

    tr_norm = tr_tensor(scheme.T; inv=inv)
    scheme.T /= tr_norm
    scheme.E1 /= norm(scheme.E1)
    scheme.E2 /= norm(scheme.E2)
    for _ in 0:sweep
        rctm_step!(scheme)
    end
    return tr_norm
end

function run!(scheme::ctm_HOTRG,
              trunc,
              criterion;
              sweep=30,
              return_cft=false,
              inv=false,
              conv_criteria=1e-12)
    area = 1
    lnz = 0.0
    cft = []

    for i in 1:criterion.n
        area *= 4.0
        tr_norm = step!(scheme, trunc; sweep=sweep, inv=inv)
        if return_cft
            push!(cft, cft_data(scheme; unitcell=2))
        end
        lnz += log(tr_norm) / area
        if abs(log(abs(tr_norm)) / area) <= conv_criteria
            @info "CTM-HOTRG converged after $i iterations!"
            break
        end
    end
    if return_cft
        return lnz, cft
    else
        return lnz
    end
end
