mutable struct c4CTM{A, S}
    T::TensorMap{A, S, 0, 4}
    C::TensorMap{A, S, 1, 1}
    E::TensorMap{A, S, 2, 1}

    function c4CTM(T::TensorMap{A, S, 0, 4}) where {A, S}
        C, E = c4CTM_init(T)

        return new{A, S}(T, C, E)
    end
end

function c4CTM(T_flipped::TensorMap{A, S, 2, 2}; symmetrize = false) where {A, S}
    T_unflipped = permute(flip(T_flipped, (1, 2); inv = true), ((), (3, 4, 2, 1)))
    if symmetrize
        T_unflipped = symmetrize_C4v(T_unflipped)
    else
        @assert norm(T_flipped - T_flipped') < 1.0e-14
        @assert norm(T_unflipped - rotl90_pf(T_unflipped)) < 1.0e-14
    end
    return c4CTM(T_unflipped)
end

function rotl90_pf(T::TensorMap{A, S, 2, 2}) where {A, S}
    return permute(T, ((3, 1), (4, 2)))
end

function rotl90_pf(T::TensorMap{A, S, 0, 4}) where {A, S}
    return permute(T, ((), (2, 3, 4, 1)))
end


function symmetrize_C4v(T_unflipped)
    T_c4_unflipped = (T_unflipped + rotl90_pf(T_unflipped) + rotl90_pf(rotl90_pf(T_unflipped)) + rotl90_pf(rotl90_pf(rotl90_pf(T_unflipped)))) / 4
    T_c4_flipped = permute(flip(T_c4_unflipped, (3, 4); inv = false), ((4, 3), (1, 2)))
    T_c4v_flipped = (T_c4_flipped + T_c4_flipped') / 2
    T_c4v_unflipped = permute(flip(T_c4v_flipped, (1, 2); inv = true), ((), (3, 4, 2, 1)))
    return T_c4v_unflipped
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

function run!(
        scheme::c4CTM, trunc::TensorKit.TruncationScheme, criterion::stopcrit;
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        S_prev = id(domain(scheme.C))

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            S = step!(scheme, trunc)

            if space(S) == space(S_prev)
                ε = norm(S^4 - S_prev^4)
            end

            S_prev = S

            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::c4CTM, trunc)
    mat, U, S = find_U_sym(scheme, trunc)

    @tensor scheme.C[-1; -2] := mat[1 2; 3 4] * U[3 4; -2] * conj(U[1 2; -1])
    @tensor scheme.E[-1 -2; -3] := scheme.E[1 5; 3] * flip(scheme.T, (3, 4); inv = true)[5 4 -2 2] *
        U[3 4; -3] *
        conj(U[1 2; -1])

    scheme.C /= norm(scheme.C)
    scheme.E /= norm(scheme.E)
    return S
end

function lnz(scheme::c4CTM)
    Z, env = tensor2env(permute(flip(scheme.T, (3, 4); inv = true), ((4, 3), (1, 2))), scheme.C, scheme.E)
    # should be inv = false ??
    return real(log(network_value(Z, env)))
end

function build_corner_matrix(scheme)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.C[1; 2] * scheme.E[-1 3; 1] *
        scheme.E[2 4; -3] *
        flip(scheme.T, 3; inv = true)[4 -4 -2 3]
    return mat
end

function find_U_sym(scheme, trunc)
    mat = build_corner_matrix(scheme)
    # avoid symmetry breaking due to numerical accuracy
    mat = 0.5 * (mat + adjoint(mat))

    U, S, _ = tsvd(mat; trunc = trunc & truncbelow(1.0e-20))
    return mat, U, S
end

function c4CTM_init(T::TensorMap{A, S, 0, 4}) where {A, S}
    S_type = scalartype(T)
    Vp = space(T)[1]'
    C = TensorMap(ones, S_type, oneunit(Vp) ← oneunit(Vp))
    E = TensorMap(ones, S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, E
end

function tensor2env(O, C, T)
    Z = InfinitePartitionFunction(O)
    env = CTMRGEnv(Z, space(C)[1])

    for i in 1:4
        env.corners[i] = C
        env.edges[i] = T
    end

    env.edges[3] = flip(T, 2)
    env.edges[4] = flip(T, 2)
    return Z, env
end

function Base.show(io::IO, scheme::c4CTM)
    println(io, "c4CTM - c4 symmetric Corner Transfer Matrix")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * E: $(summary(scheme.E))")
    return nothing
end
