using TensorKit, MPSKit, PEPSKit

mutable struct c4CTM
    T::AbstractTensorMap
    C::AbstractTensorMap
    E::AbstractTensorMap
end

c4CTM(T) = c4CTM(T, T, T)

function run!(scheme::c4CTM, trunc::TensorKit.TruncationScheme)
    C, E = CTM(scheme.T, trunc)
    scheme.C = C
    scheme.E = E
    scheme.lnz = lnz(scheme)
    return scheme
end

function lnz(scheme::c4CTM)
    Z, env = tensor2env(scheme.T, scheme.C, scheme.E)
    return real(log(network_value(Z, env)))
end

# Below, I wrote a code with the following correspondence. (O,C,T) <=> (scheme.T, scheme.C, scheme.E)
# https://www.issp.u-tokyo.ac.jp/public/caqmp2019/slides/808L_Okubo.pdf
#=
┌───────┐       ┌───────┐       
│       │       │       │       
│       │       │       │       
│   C   ├──────►│   E   ├──────►
│       │       │       │       
└───────┘       └───────┘       
    ▲               ▲           
    │               │           
    │               │           
=#

function flip_Vphy(A)
    sp = space(A)
    return TensorMap(A.data, sp[1] ⊗ sp[2]' ← sp[3]')
end

function build_corner_matrix(O, C, T)
    @tensoropt mat[-1 -2; -3 -4] :=
        C[1; 2] * flip_Vphy(T)[-1 3; 1] * T[2 4; -3] * O[3 -2; 4 -4]
    return mat
end

function find_U_sym(O, C, T, trunc; return_mat = true, symmetrize = true)
    mat = build_corner_matrix(O, C, T)
    # avoid the symmetry breaking due to the numerical accuracy
    if symmetrize
        mat = 0.5 * (mat + adjoint(mat))
    end
    # if !ishermitian(mat)
    #     @error "Corner matrix is not hermitian"
    # end
    U, S, Vt = tsvd(mat; trunc = trunc & truncbelow(1e-20))
    if return_mat
        return mat, U, S
    else
        return U
    end
end

function update_CTM(O, C, T, trunc; symmetrize = true)
    mat, U, S = find_U_sym(O, C, T, trunc; symmetrize = symmetrize)
    @tensoropt Cnew[-1; -2] := mat[1 2; 3 4] * U[3 4; -2] * conj(U[1 2; -1])
    @tensoropt Tnew[-1 -2; -3] := T[1 5; 3] * O[2 -2; 5 4] * U[3 4; -3] * conj(U[1 2; -1])
    S /= abs(tr(S^4))^0.25
    Cnew /= abs(tr(Cnew^4))^0.25
    Tnew /= norm(Tnew.data)
    return Cnew, Tnew, S
end

function initialize_CT(O)
    elt = typeof(O.data[1])
    Vp = space(O)[3]'
    C = TensorMap(ones, elt, oneunit(Vp) ← oneunit(Vp))
    T = TensorMap(ones, elt, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, T
end

function CTM(
    O,
    trunc;
    maxiter = 1e4,
    tol = 1e-12,
    return_hist = false,
    return_PEPSKit = false,
    initial_CT = nothing,
    symmetrize = true,
)
    if initial_CT == nothing
        C, T = initialize_CT(O)
    else
        C, T = initial_CT
    end
    ϵ = 1.0
    S = C
    iter = 1
    flag = false
    ϵ_list = []
    while iter < maxiter && ϵ > tol
        C, T, S_next = update_CTM(O, C, T, trunc; symmetrize = symmetrize)
        if space(S) == space(S_next) && iter > 5
            ϵ = norm(S - S_next)
            flag = true
        end
        S = S_next
        @debug "iteration :$(iter) \t ϵ :$ϵ"
        iter += 1
        if flag
            push!(ϵ_list, ϵ)
        end
    end
    if iter == maxiter
        @info "maxiteration has been reached"
    else
        @info "Converged"
    end
    @info "iteration: $(iter) \t error: $(ϵ)"
    if return_hist
        return C, T, ϵ_list
    end
    return C, T
end

function tensor2env(O, C, T)
    Z = InfinitePartitionFunction(O;)
    env = CTMRGEnv(Z, space(C)[1])
    for i = 1:4
        env.corners[i] = C
        env.edges[i] = T
    end

    env.edges[3] = flip_Vphy(T)
    env.edges[4] = flip_Vphy(T)
    return Z, env
end
