function Φ_action(left::AbstractTensorMap{E, S, 2, 2}, right::AbstractTensorMap{E, S, 2, 2}, x::AbstractTensorMap{E, S, 2, 2}) where {E, S}
    @plansor contractcheck = true Φx[1 6; 3 8] := left[1 2; 3 4] * x[2 5; 4 7] * right[5 6; 7 8]
    return Φx
end

function ΨBΨB_conj(psiB::Vector{<:AbstractTensorMap{E, S, 1, 2}}) where {E, S}
    @assert iseven(length(psiB))
    return map(eachindex(psiB)) do i_conj
        if iseven(i_conj)
            i = i_conj - 1
            return @plansor BB[-1 -2; -3 -4] := psiB[i][1; -4 -2] * psiB[i]'[-3 -1; 1]
        else
            i = i_conj + 1
            return @plansor BB[-1 -2; -3 -4] := psiB[i][-4; -2 1] * psiB[i]'[-1 1; -3]
        end
    end
end

function ΨBΨA_conj(psiB::Vector{<:AbstractTensorMap{E, S, 1, 2}}, psiA::Vector{<:AbstractTensorMap{E, S, 1, 3}}) where {E, S}
    @assert length(psiB) == 2 * length(psiA)
    return map(eachindex(psiA)) do i
        return @plansor BBA[-1 -2; -3 -4] :=  psiB[2 * i]'[-1 1; 3] *
            psiA[i][2; -4 -2 1] * psiB[2 * i - 1]'[-3 3; 2]
    end
end

function Φ_cost(psiB::Vector{<:AbstractTensorMap{E, S, 1, 2}}, psiA::Vector{<:AbstractTensorMap{E, S, 1, 3}}) where {E, S}
    NA = length(psiA)
    NA_half = NA ÷ 2

    psiApsiA = ΨAΨA(psiA)
    psiBpsiB = ΨBΨB(psiB)
    psiBpsiA = ΨBΨA(psiB, psiA)

    psiApsiA_conj = psiApsiA[[(NA ÷ 2 + 1) : NA..., 1 : NA ÷ 2...]]
    psiBpsiB_conj = ΨBΨB_conj(psiB)
    psiBpsiA_conj = ΨBΨA_conj(psiB, psiA)

    AAs = Float64[]

    right_AA = id(E, codomain(psiApsiA[1]))
    left_AA = id(E, codomain(psiApsiA_conj[1]))

    cache_AA = right_cache(psiApsiA)
    cache_AA_conj = right_cache(psiApsiA_conj)

    for i in 1 : (NA ÷ 2)
        left = cache_AA_conj[i] * left_AA
        right = transpose(cache_AA[i] * right_AA)
        TA = transpose(psiA[i], ((2, 1), (3, 4)))
        AA_temp = real(tr(TA' * Φ_action(left, right, TA)))
        push!(AAs, AA_temp)
        left_AA = left_AA * psiApsiA_conj[i]
        right_AA = right_AA * psiApsiA[i]
    end

    costs = Float64[]

    cache_BB = right_cache(psiBpsiB)
    cache_BB_conj = right_cache(psiBpsiB_conj)
    cache_BA = right_cache(psiBpsiA)
    cache_BA_conj = right_cache(psiBpsiA_conj)

    right_BB = id(E, codomain(psiBpsiB[1]))
    left_BB = id(E, codomain(psiBpsiB_conj[1]))
    
    right_BA = id(E, codomain(psiBpsiA[1]))
    left_BA = id(E, codomain(psiBpsiA_conj[1]))

    for i in 1 : NA
        Φ_left_BB = cache_BB_conj[2 * i] * left_BB
        Φ_right_BB = transpose(cache_BB[2 * i] * right_BB)

        Φ_left_BA = cache_BA_conj[i] * left_BA
        Φ_right_BA = transpose(cache_BA[i] * right_BA)

        @plansor TBB[-1 -2; -3 -4] := psiB[2 * i - 1][-2; -1 m] * psiB[2 * i][m; -3 -4]
        TA = transpose(psiA[i], ((2, 1), (3, 4)))

        BB = real(tr(TBB' * Φ_action(Φ_left_BB, Φ_right_BB, TBB)))
        BA = tr(TBB' * Φ_action(Φ_left_BA, Φ_right_BA, TA))
        AA = AAs[mod(i - 1, NA_half) + 1]

        cost_this = (BB + AA - 2 * real(BA)) / AA
        push!(costs, cost_this)

        left_BB = left_BB * psiBpsiB_conj[2 * i - 1] * psiBpsiB_conj[2 * i]
        right_BB = right_BB * psiBpsiB[2 * i - 1] * psiBpsiB[2 * i]
        left_BA = left_BA * psiBpsiA_conj[i]
        right_BA = right_BA * psiBpsiA[i]
    end
    
    return costs
end