# C4 & inversion symmetric LoopTNR described in Fig. S6 of Phys. Rev. Lett. 118, 110504 (2017)
mutable struct SLoopTNR <: TNRScheme
    T::TensorMap

    gradalg::OptimKit.LBFGS
    finalize!::Function
    function SLoopTNR(T::TensorMap;
                      gradalg=LBFGS(10; verbosity=0, gradtol=6e-7, maxiter=40000),
                      finalize=(finalize!))
        return new(T, gradalg, finalize)
    end
end

########## Initial tensor ##########
function classical_ising_inv(β)
    x = cosh(β)
    y = sinh(β)

    S = ℤ₂Space(0 => 1, 1 => 1)
    T = zeros(Float64, S ⊗ S ← S' ⊗ S')
    block(T, Irrep[ℤ₂](0)) .= [2x^2 2x*y; 2x*y 2y^2]
    block(T, Irrep[ℤ₂](1)) .= [2x*y 2x*y; 2x*y 2x*y]

    return permute(T, (1, 2, 3, 4))
end
classical_ising_inv() = classical_ising_inv(ising_βc)

########## utility functions ##########
function trnorm_2x2(T)
    @tensor TT[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(T[1 -2 2 -4])
    return sqrt(TTtoNorm(TT))
end

########## Cost function ##########
function StoSS(S)
    V = domain(S)[1]
    b = isomorphism(V, V')
    @tensor SS[-1 -2 -3 -4] := S[-1 -2; 1] * S[-3 -4; 2] * b[1 2]
    return SS
end

function TTtoNorm(TT)
    V = domain(TT)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    TTb = TT * b
    @tensor T4[-1 -2; -3 -4] := TT[-1 -2; 1 2] * TTb[-3 -4; 1 2]
    V = domain(T4)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    T4b = T4 * b
    @tensor T8[-1 -2; -3 -4] := T4[-1 -2; 1 2] * T4b[-3 -4; 1 2]
    V = domain(T8)
    b = isomorphism(V[1] ⊗ V[2], V[1]' ⊗ V[2]')
    return tr(T8 * b)
end

function cost_looptnr(S, T)
    @assert eltype(S) == Float64 "Modification is needed for complex numbers!"
    SS = StoSS(S)
    @tensor TT[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(T[1 -2 2 -4])
    @tensor TSS[-1 -2; -3 -4] := T[1 -1 2 -3] * conj(SS[1 -2 2 -4])
    @tensor S4[-1 -2; -3 -4] := SS[1 -1 2 -3] * conj(SS[1 -2 2 -4])
    # T
    return TTtoNorm(TT) + TTtoNorm(S4) - 2 * TTtoNorm(TSS)
end

########## Gradient Optimization ##########
function fg(f, A)
    f_out, g = Zygote.withgradient(f, A)

    return f_out, g[1]
end

function optimize_S(scheme, S)
    opt_fun(x) = cost_looptnr(x, scheme.T)
    opt_fg(x) = fg(opt_fun, x)
    Sopt, fx, gx, numfg, normgradhistory = optimize(opt_fg, S,
                                                    scheme.gradalg)
    return Sopt
end

########## Entanglement filtering ##########
function Ψ_center(T)
    Tflip = flip(T, (1, 2, 3, 4))
    psi = AbstractTensorMap[permute(T, ((2,), (1, 3, 4))),
                            permute(Tflip, ((4,), (3, 1, 2))),
                            permute(T, ((2,), (1, 3, 4))),
                            permute(Tflip, ((4,), (3, 1, 2)))]
    return psi
end

function Ψ_corner(T)
    Tflip = flip(T, (1, 2, 3, 4))
    psi = AbstractTensorMap[permute(T, ((3,), (4, 2, 1))),
                            permute(Tflip, ((1,), (2, 4, 3))),
                            permute(T, ((3,), (4, 2, 1))),
                            permute(Tflip, ((1,), (2, 4, 3)))]
    return psi
end

function entanglement_filtering(T; trunc=truncbelow(1e-12))
    entanglement_function(steps, data) = abs(data[end])
    entanglement_criterion = maxiter(100) & convcrit(1e-12, entanglement_function)

    psi_center = Ψ_center(T)
    psi_corner = Ψ_corner(T)

    PR_list, PL_list = TNRKit.find_projectors(psi_center, entanglement_criterion, trunc)
    P_bottom = PL_list[1]
    P_right = PL_list[1]

    PR_list, PL_list = TNRKit.find_projectors(psi_corner, entanglement_criterion, trunc)
    P_top = PL_list[3]
    P_left = PL_list[3]

    @tensor T_new[-1 -2 -3 -4] := T[1 2 3 4] * P_left[-1; 1] * P_bottom[-2; 2] *
                                  P_top[-3; 3] * P_right[-4; 4]
    return T_new
end

########## Initialization of loop optimizations ##########
function decompose_T(T, trunc)
    u, s, vt = tsvd(T, (1, 2), (3, 4); trunc)
    return u * sqrt(s)
end

function ef_oneloop(T, trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_center(T)
    ΨB = []

    for i in 1:4
        s1, s2 = TNRKit.SVD12(ΨA[i], truncdim(trunc.dim * 2))
        push!(ΨB, s1)
        push!(ΨB, s2)
    end

    ΨB_function(steps, data) = abs(data[end])
    criterion = maxiter(100) & convcrit(1e-12, ΨB_function)
    PR_list, PL_list = TNRKit.find_projectors(ΨB, criterion, trunc)

    ΨB_disentangled = []
    for i in 1:1
        @tensor B1[-2 -1; -3] := ΨB[i][-1; -2 2] *
                                 PR_list[mod(i, 8) + 1][2; -3]
        push!(ΨB_disentangled, B1)
    end
    S = ΨB_disentangled[1]
    return S
end

########## Updating the tensor ##########
function combine_4S(S)
    Sflip = flip(S, (1, 2))
    @tensor Tnew[-1 -2 -3 -4] := S[1 2; -4] * Sflip[1 4; -3] * S[3 4; -1] * Sflip[3 2; -2]
    return Tnew
end

########## Main funcitons ##########
function step!(scheme, trunc, oneloop)
    scheme.T = entanglement_filtering(scheme.T)
    if oneloop == true
        S = ef_oneloop(scheme.T, trunc)
    else
        S = decompose_T(scheme.T, trunc)
    end
    S = optimize_S(scheme, S)
    scheme.T = combine_4S(S)
    return scheme
end

function run!(scheme::SLoopTNR, trscheme::TensorKit.TruncationScheme,
              criterion::TNRKit.stopcrit; finalize_beginning=true, oneloop=true,
              verbosity=1)
    data = []

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"

        if finalize_beginning
            push!(data, scheme.finalize!(scheme))
        end
        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme, oneloop)
            push!(data, scheme.finalize!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end
        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end

function Base.show(io::IO, scheme::SLoopTNR)
    println(io, "Symmetric LoopTNR - C4 and reflection symmetric scheme")
    println(io, "  * T: $(summary(scheme.T))")
    println(io, "  * gradalg: $(summary(scheme.gradalg))")
    return nothing
end
