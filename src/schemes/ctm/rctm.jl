#=
CTM with spatial reflection symmetry along x and y axis. This allows the different edge tensors for the vertical and horizontal edges.
The corner tensors are related by its mirror images.

    ┌──┐    ┌────┐
    │  │    │    │
───►│E1│───►│ C2 │
    └──┘    └──┬─┘
      ▲        │  
      │        ▼  
             ┌──┐ 
             │  │ 
          ──►│E2│ 
             └─┬┘ 
               │  
               ▼  
=#
mutable struct rCTM{A, S}
    T::TensorMap{A, S, 2, 2}
    C2::TensorMap{A, S, 1, 1}
    E1::TensorMap{A, S, 2, 1}
    E2::TensorMap{A, S, 2, 1}
    function rCTM(T::TensorMap{A, S, 2, 2}) where {A, S}
        if typeof(T.data[1]) != Float64
            @error "This scheme only support tensors with real numbers"
        end
        C, E1, E2 = rCTM_init(T)
        @assert BraidingStyle(sectortype(T)) == Bosonic() "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for rCTM"
        return new{A, S}(T, C, E1, E2)
    end
end

function rCTM_init(T)
    elt = typeof(T.data[1])
    Vp1 = space(T)[3]'
    Vp2 = space(T)[4]'
    C = TensorMap(ones, elt, oneunit(Vp1) ← oneunit(Vp2))
    E1 = TensorMap(ones, elt, oneunit(Vp1) ⊗ Vp1 ← oneunit(Vp1))
    E2 = TensorMap(ones, elt, oneunit(Vp2) ⊗ Vp2 ← oneunit(Vp2))
    return C, E1, E2
end

function rt_build_corner_matrix(scheme::rCTM)
    @tensor opt = true mat[-1 -2; -3 -4] := scheme.E1[-1 3; 1] * scheme.C2[1; 2] *
        scheme.E2[2 4; -3] * scheme.T[-2 -4; 3 4]
    return mat
end

function find_UVt(scheme, trunc)
    mat = rt_build_corner_matrix(scheme)
    U, S, Vt = tsvd(mat; trunc = trunc & truncbelow(1.0e-20))
    return mat, U, S, Vt
end

function step!(scheme::rCTM, trunc)
    mat, U, S, Vt = find_UVt(scheme, trunc)

    scheme.C2 = adjoint(U) * mat * adjoint(Vt)
    @tensor opt = true scheme.E1[-1 -2; -3] := scheme.E1[1 5; 3] * scheme.T[2 -2; 5 4] *
        U[3 4; -3] * conj(U[1 2; -1])
    @tensor opt = true scheme.E2[-1 -2; -3] := scheme.E2[1 5; 3] * scheme.T[-2 4; 2 5] *
        conj(Vt[-3; 3 4]) * Vt[-1; 1 2]

    scheme.C2 /= norm(scheme.C2)
    scheme.E1 /= norm(scheme.E1)
    scheme.E2 /= norm(scheme.E2)
    # symmetrization of the edges to avoid the breaking of reflection due to the numerical error
    scheme.E1 = (scheme.E1 + flip(permute(scheme.E1, ((3, 2), (1,))), (1, 3)) / 2)
    scheme.E2 = (scheme.E2 + flip(permute(scheme.E1, ((3, 2), (1,))), (1, 3)) / 2)
    return S
end

function run!(
        scheme::rCTM,
        trunc::TensorKit.TruncationScheme,
        criterion::TNRKit.stopcrit;
        verbosity = 1,
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        S_prev = id(domain(scheme.C2))

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

        @infov 1 "Simulation finished\n $(TNRKit.stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function lnz(scheme::rCTM)
    Z, env = tensor2env(scheme.T, scheme.C2, scheme.E1, scheme.E2)
    return real(log(network_value(Z, env)))
end

function tensor2env(T, C2, E1, E2)
    Z = InfinitePartitionFunction(T)
    env = CTMRGEnv(Z, space(C2)[1])
    for i in 1:2
        env.corners[2 * i] = C2
        env.edges[2 * i] = E2
        env.corners[2 * i - 1] = adjoint(C2)
        env.edges[2 * i - 1] = E1
    end
    env.edges[3] = flip(env.edges[3], 2)
    env.edges[4] = flip(env.edges[4], 2)
    return Z, env
end

function Base.show(io::IO, scheme::rCTM)
    println(io, "rCTM - reflection symmetric Corner Transfer Matrix")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * E1: $(summary(scheme.E1))")
    println(io, "  * E2: $(summary(scheme.E2))")
    println(io, "  * C: $(summary(scheme.C2))")
    return nothing
end
