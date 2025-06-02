mutable struct c4CTM{A,S}
    T::TensorMap{A,S,2,2}
    C::TensorMap{A,S,1,1}
    E::TensorMap{A,S,2,1}

    function c4CTM(T::TensorMap{A,S,2,2}) where {A,S}
        C, E = c4CTM_init(T)

        @assert BraidingStyle(sectortype(T)) == Bosonic() "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for c4CTM"
        return new{A,S}(T, C, E)
    end
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

function run!(scheme::c4CTM, trunc::TensorKit.TruncationScheme, criterion::Stopcrit;
              verbosity=1)
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
    @tensor scheme.E[-1 -2; -3] := scheme.E[1 5; 3] * scheme.T[2 -2; 5 4] *
                                   U[3 4; -3] *
                                   conj(U[1 2; -1])

    scheme.C /= norm(scheme.C)
    scheme.E /= norm(scheme.E)
    return S
end

function lnz(scheme::c4CTM)
    Z, env = tensor2env(scheme.T, scheme.C, scheme.E)
    return real(log(network_value(Z, env)))
end

flip_Vphy(A) = flip(A, 2)

function build_corner_matrix(scheme)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.C[1; 2] * flip_Vphy(scheme.E)[-1 3; 1] *
                                            scheme.E[2 4; -3] *
                                            scheme.T[3 -2; 4 -4]
    return mat
end

function find_U_sym(scheme, trunc)
    mat = build_corner_matrix(scheme)
    # avoid symmetry breaking due to numerical accuracy
    mat = 0.5 * (mat + adjoint(mat))

    U, S, _ = tsvd(mat; trunc=trunc & truncbelow(1e-20))
    return mat, U, S
end

function c4CTM_init(T::TensorMap{A,S,2,2}) where {A,S}
    S_type = scalartype(T)
    Vp = space(T)[3]'
    C = TensorMap(ones, S_type, oneunit(Vp) ← oneunit(Vp))
    E = TensorMap(ones, S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, E
end

function tensor2env(O, C, T)
    Z = InfinitePartitionFunction(O;)
    env = CTMRGEnv(Z, space(C)[1])

    for i in 1:4
        env.corners[i] = C
        env.edges[i] = T
    end

    env.edges[3] = flip_Vphy(T)
    env.edges[4] = flip_Vphy(T)
    return Z, env
end

function Base.show(io::IO, scheme::c4CTM)
    println(io, "c4CTM - c4 symmetric Corner Transfer Matrix")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * C: $(summary(scheme.C))")
    println(io, "  * E: $(summary(scheme.E))")
    return nothing
end
