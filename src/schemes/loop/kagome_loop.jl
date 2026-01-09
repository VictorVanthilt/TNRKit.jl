mutable struct KagomeLoopTNR{E, S, TT <: AbstractTensorMap{E, S, 2, 2}} <: LinearLoopScheme{E, S}
    "Central tensor on sublattice A"
    TA::TT

    "Central tensor on sublattice B"
    TB::TT

    "Central tensor on sublattice A"
    TC::TT

    function KagomeLoopTNR(TA::TT, TB::TT, TC::TT) where {E, S, TT <: AbstractTensorMap{E, S, 2, 2}}
        return new{E, S, TT}(TA, TB, TC)
    end
end

function Ψ_A(TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, TC::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    ΨA = [
        transpose(TA, ((1,), (3, 4, 2)); copy = true),
        transpose(TB, ((3,), (4, 2, 1)); copy = true),
        transpose(TC, ((3,), (4, 2, 1)); copy = true),
        transpose(TA, ((4,), (2, 1, 3)); copy = true),
        transpose(TB, ((2,), (1, 3, 4)); copy = true),
        transpose(TC, ((2,), (1, 3, 4)); copy = true),
    ]
    return ΨA
end
function Ψ_A(scheme::KagomeLoopTNR)
    return Ψ_A(scheme.TA, scheme.TB, scheme.TC)
end

function _entanglement_filtering(
        TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, TC::AbstractTensorMap{E, S, 2, 2},
        entanglement_criterion::stopcrit, trunc::TensorKit.TruncationScheme
    ) where {E, S}
    ΨA = Ψ_A(TA, TB, TC)
    PRs, PLs = find_projectors(
        ΨA, [1, 1, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3],
        entanglement_criterion, trunc
    )

    @plansor TA[-1 -2; -3 -4] := TA[1 2; 3 4] * PLs[1][-1; 1] * PRs[2][2; -2] * PLs[4][-4; 4] * PRs[5][3; -3]
    @plansor TB[-1 -2; -3 -4] := TB[1 2; 3 4] * PLs[2][-3; 3] * PRs[3][1; -1] * PLs[5][-2; 2] * PRs[6][4; -4]
    @plansor TC[-1 -2; -3 -4] := TC[1 2; 3 4] * PLs[3][-3; 3] * PRs[4][1; -1] * PLs[6][-2; 2] * PRs[1][4; -4]
    return TA, TB, TC
end


function entanglement_filtering!(
        scheme::KagomeLoopTNR,
        trunc::TensorKit.TruncationScheme,
        entanglement_criterion::stopcrit = default_entanglement_criterion
    )
    scheme.TA, scheme.TB, scheme.TC = _entanglement_filtering(
        scheme.TA, scheme.TB, scheme.TC, entanglement_criterion, trunc
    )
    return scheme
end

function ΨB_to_TATBTC(psiB::Vector{T}, TA::AbstractTensorMap{E, S, 2, 2}, TB::AbstractTensorMap{E, S, 2, 2}, TC::AbstractTensorMap{E, S, 2, 2}) where {T <: AbstractTensorMap{<:Any, <:Any, 1, 2}, E, S}
    @plansor opt = true newTA[-1 -2; -3 -4] := psiB[12][-1; B12 121] * psiB[1][121; B1 -2] * TB[B12 B1; B7 B6] * psiB[6][-4; B6 67] * psiB[7][67; B7 -3]
    @plansor opt = true newTB[-1 -2; -3 -4] := psiB[2][-3; C2 23] * psiB[3][23; C3 -1] * TC[C2 C3; C9 C8] * psiB[8][-2; C8 89] * psiB[9][89; C9 -4]
    @plansor opt = true newTC[-1 -2; -3 -4] := psiB[4][-3; A4 45] * psiB[5][45; A5 -1] * TA[A5 A10; A4 A11] * psiB[10][-2; A10 1011] * psiB[11][1011; A11 -4]
    return newTA, newTB, newTC
end

function loop_opt!(
        scheme::KagomeLoopTNR,
        loop_criterion::stopcrit,
        trunc::TensorKit.TruncationScheme,
        truncentanglement::TensorKit.TruncationScheme,
        verbosity::Int
    )
    psiA = Ψ_A(scheme)
    psiB = loop_opt(psiA, loop_criterion, trunc, truncentanglement, verbosity)
    scheme.TA, scheme.TB, scheme.TC = ΨB_to_TATBTC(psiB, scheme.TA, scheme.TB, scheme.TC)
    return scheme
end

function Base.show(io::IO, scheme::KagomeLoopTNR)
    println(io, "KagomeLoopTNR - Loop Tensor Network Renormalization on Kagome lattice")
    println(io, "  * TA: $(summary(scheme.TA))")
    println(io, "  * TB: $(summary(scheme.TB))")
    println(io, "  * TC: $(summary(scheme.TC))")
    return nothing
end
