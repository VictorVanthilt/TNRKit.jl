mutable struct GBTRG <: TNRScheme
    TA::TensorMap
    TB::TensorMap
    S1::TensorMap
    S2::TensorMap
    S3::TensorMap
    S4::TensorMap
    k::Float64

    finalize!::Function
    function GBTRG(TA::TensorMap{E,S,2,2}, TB::TensorMap{E,S,2,2}, k::Number;
                   finalize=(finalize!)) where {E,S}
        # Construct S1 and S2 as identity matrices.
        return new(TA, TB, id(space(TA, 2)), id(space(TA, 1)), id(space(TB, 2)),
                   id(space(TB, 2)), k, finalize)
    end
    function GBTRG(T::TensorMap{E,S,2,2}, k::Number;
                   finalize=(finalize!)) where {E,S}
        # Construct S1 and S2 as identity matrices.
        return new(T, T, id(space(T, 2)), id(space(T, 1)), id(space(T, 2)),
                   id(space(T, 2)), k, finalize)
    end
end

GBTRG(T::TensorMap; kwargs...) = GBTRG(T, -0.5; kwargs...)
GBTRG(TA::TensorMap, TB::TensorMap; kwargs...) = GBTRG(TA, TB, -0.5; kwargs...)

function pseudopow(t::DiagonalTensorMap, a::Real; tol=eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for I in LinearAlgebra.diagind(b)
            b[I] = b[I] < tol ? b[I] : b[I]^a
        end
    end
    return t′
end

#Initialisation of Ψ_A

function Ψ_A(scheme::GBTRG)
    psi = AbstractTensorMap[permute(scheme.TA, ((2,), (1, 3, 4))),
                            scheme.S4,
                            permute(scheme.TB, ((1,), (3, 4, 2))),
                            permute(scheme.S3, ((2,), (1,))),
                            permute(scheme.TA, ((3,), (4, 2, 1))),
                            permute(scheme.S2, ((2,), (1,))),
                            permute(scheme.TB, ((4,), (2, 1, 3)))
                            scheme.S1]
    return psi
end

#Entanglement Filtering 
entanglement_function(steps, data) = abs(data[end])
entanglement_criterion = maxiter(100) & convcrit(1e-15, entanglement_function)

loop_criterion = maxiter(50) & convcrit(1e-8, entanglement_function)

function entanglement_filtering!(scheme::GBTRG, entanglement_criterion::stopcrit,
                                 trunc::TensorKit.TruncationScheme)
    ΨA = Ψ_A(scheme)
    PR_list, PL_list = find_projectors(ΨA, entanglement_criterion, trunc)

    TA = copy(scheme.TA)
    TB = copy(scheme.TB)
    S1 = copy(scheme.S1)
    S2 = copy(scheme.S2)
    S3 = copy(scheme.S3)
    S4 = copy(scheme.S4)

    @tensor scheme.TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PR_list[6][1; -1] *
                                       PL_list[1][-2; 2] * PR_list[2][4; -4] *
                                       PL_list[5][-3; 3]
    @tensor scheme.TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PL_list[3][-1; 1] *
                                       PR_list[4][2; -2] * PL_list[7][-4; 4] *
                                       PR_list[8][3; -3]

    @tensor scheme.S1[-1; -2] := S1[1; 2] * PL_list[8][-1; 1] * PR_list[1][2; -2]
    @tensor scheme.S2[-1; -2] := S2[1; 2] * PL_list[6][-2; 2] * PR_list[7][1; -1]
    @tensor scheme.S3[-1; -2] := S3[1; 2] * PL_list[4][-2; 2] * PR_list[5][1; -1]
    @tensor scheme.S4[-1; -2] := S4[1; 2] * PL_list[2][-1; 1] * PR_list[3][2; -2]

    return scheme
end

function entanglement_filtering!(scheme::LoopTNR, trunc::TensorKit.TruncationScheme)
    return entanglement_filtering!(scheme, entanglement_criterion, trunc)
end



function Base.show(io::IO, scheme::GBTRG)
    println(io, "LoopTNR - Loop Tensor Network Renormalization")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    return nothing
end
