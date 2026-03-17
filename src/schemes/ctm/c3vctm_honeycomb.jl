"""
$(TYPEDEF)

Corner Transfer Matrix Renormalization Group for the honeycomb lattice

### Constructors
    $(FUNCTIONNAME)(T)
    $(FUNCTIONNAME)(T, [, symmetrize=false])

```
     (120°)
        ╲ 
         ╲ 
          ╲ 
           T -----(0°)
           ╱
          ╱
         ╱
      (240°)
```

CTM can be called with a (2, 1) tensor, where the directions are (240°, 0°, 120°) clockwise with respect to the positive x-axis.
In the flipped arrow convention, the arrows point from (120°) to (240°, 0°).
or with a (0,3) tensor (120°, 0°, 240°) where all arrows point inward (unflipped arrow convention).
The keyword argument symmetrize makes the tensor C6v symmetric when set to true. If symmetrize = false, it checks the symmetry explicitly.

### Running the algorithm
    run!(::CTM, trunc::TruncationStrategy, stop::Stopcrit[, verbosity=1])

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Lukin et al. Phys. Rev. B 107.054424 (2023)](@cite lukin2023)
"""
mutable struct c3vCTM_honeycomb{A, S}
    T::TensorMap{A, S, 0, 3}
    C::TensorMap{A, S, 1, 1}
    L::TensorMap{A, S, 2, 1}
    R::TensorMap{A, S, 2, 1}

    function c3vCTM_honeycomb(T::TensorMap{A, S, 0, 3}) where {A, S}
        C, Ea, Eb = c3vCTM_honeycomb_init(T)

        if BraidingStyle(sectortype(T)) != Bosonic()
            @warn "$(summary(BraidingStyle(sectortype(T)))) braiding style is not supported for c6vCTM"
        end
        return new{A, S}(T, C, Ea, Eb)
    end
end

function c3vCTM_honeycomb(T_flipped::TensorMap{A, S, 2, 1}; symmetrize = false) where {A, S}
    T_unflipped = permute(flip(T_flipped, [1 2]; inv = true), ((), (3, 2, 1)))
    if symmetrize
        T_unflipped = symmetrize_C6v_honeycomb(T_unflipped)
    else
        @assert norm(T_unflipped - rotl120_pf_honeycomb(T_unflipped)) < 1.0e-14 "Tensor is not C6 symmetric. Error = $(norm(T_unflipped - rotl60_pf(T_unflipped)))"
    end

    return c3vCTM_honeycomb(T_unflipped)
end

function c3vCTM_honeycomb_init(T::TensorMap{A, S, 0, 3}) where {A, S}
    S_type = scalartype(T)
    Vp = space(T)[1]'
    C = ones(S_type, oneunit(Vp) ← oneunit(Vp))
    L = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    R = ones(S_type, oneunit(Vp) ⊗ Vp ← oneunit(Vp))
    return C, L, R
end

# Functions to permute (flipped and unflipped) tensors under 60 degree rotation
function rotl120_pf_honeycomb(T::TensorMap{A, S, 2, 1}) where {A, S}
    return permute(T, ((3, 1), (2,)))
    return permute(T, ((4, 1, 2), (5, 6, 3)))
end

function rotl120_pf_honeycomb(T::TensorMap{A, S, 0, 3}) where {A, S}
    return permute(T, ((), (2, 3, 1)))
end

function symmetrize_C6v_honeycomb(T_unflipped)
    return (T_unflipped + rotl120_pf_honeycomb(T_unflipped) + rotl120_pf_honeycomb(rotl120_pf_honeycomb(T_unflipped))) / 3
end

# Based on
# https://arxiv.org/abs/2209.03428

function run!(
        scheme::c3vCTM_honeycomb, trunc::TruncationStrategy, criterion::stopcrit;
        verbosity = 1
    )
    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        steps = 0
        crit = true
        ε = Inf
        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), ε = $(ε)"

            ε = step!(scheme, trunc)
            steps += 1
            crit = criterion(steps, ε)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, ε))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return lnz(scheme)
end

function step!(scheme::c3vCTM_honeycomb, trunc)
    D, W = calculate_projectors(scheme, trunc)

    renormalize_corners!(scheme, W)
    scheme.C /= norm(scheme.C)

    renormalize_edges!(scheme, W)
    scheme.L /= norm(scheme.R)
    scheme.R /= norm(scheme.R)
    return error_measure(scheme)
end

function calculate_projectors(scheme, trunc)
    @tensor opt = true mat[χout Dout; χin Din] := scheme.L[χout DNW; χN] * scheme.C[χN χNE] * scheme.R[χNE DE; χin] *
        scheme.T[DNW DE DC] * conj(scheme.T[Din DC Dout])

    D, W = eig_trunc(mat; trunc)
    return D, W
end

function renormalize_corners!(scheme, W)
    return @tensor opt = true scheme.C[χout; χin] := scheme.L[χWc DNW; χN] * scheme.C[χN χNE] * scheme.R[χNE DE; χE] *
        scheme.T[DNW DE DC] * conj(flip(scheme.T, 1)[Din Dout DC]) *
        W[χE Din; χin] * conj(W[χWc Dout; χout])
end

function renormalize_edges!(scheme, W)
    @tensor opt = true scheme.L[χout Dout; χin] := scheme.L[χ1 D1; χ2] * scheme.T[D1 D2 D3] * conj(scheme.T[Dout D4 D3]) *
        W[χ2 D2; χin] * conj(W[χ1 D4; χout])
    return @tensor opt = true scheme.R[χout Dout; χin] := flip(scheme.R, 2)[χ1 D1; χ2] * conj(scheme.T[D2 D3 D1]) * flip(scheme.T, 3)[D2 D4 Dout] *
        W[χ2 D4; χin] * conj(W[χ1 D3; χout])
end

function error_measure(scheme)
    @tensor LHS[χout Dout; χin] := scheme.C[χout; χ] * scheme.L[χ Dout; χin]
    @tensor RHS[χout Dout; χin] := scheme.R[χout Dout; χ] * scheme.C[χ; χin]
    return norm(LHS - RHS)
end

function lnz(scheme::c3vCTM_honeycomb)
    return real(log(network_value(scheme)))
end

function network_value(scheme::c3vCTM_honeycomb)
    nw_corners = _contract_corners(scheme)
    nw_full = _contract_site_large(scheme)
    nw_L = _contract_edges_L(scheme)
    nw_R = _contract_edges_R(scheme)
    return sqrt(nw_full^3 * nw_corners / ((nw_L * nw_R)^2))
end

function _contract_corners(scheme::c3vCTM_honeycomb)
    return @tensor scheme.C[1; 2] * scheme.C[2; 3] * scheme.C[3; 4] * scheme.C[4; 5] * scheme.C[5; 6] * scheme.C[6; 1]
end

function _contract_site_large(scheme::c3vCTM_honeycomb)
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.L[χNL DNW; χNR] *
        scheme.C[χNR χNEL] * scheme.R[χNEL DE; χNER] *
        scheme.C[χNER; χSE] * scheme.C[χSE; χSL] * scheme.L[χSL DSE; χSR] *
        scheme.C[χSR; χSWL] * scheme.R[χSWL DW; χSWR] *
        scheme.T[DNW DE DC] * conj(flip(scheme.T, [1 2])[DSE DW DC])
end

function _contract_edges_L(scheme::c3vCTM_honeycomb)
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.L[χNL DNW; χNR] *
        scheme.C[χNR χNE] * scheme.C[χNE; χSEL] * scheme.L[χSEL DE; χSER] *
        scheme.C[χSER; χS] * scheme.C[χS; χSWL] * scheme.L[χSWL DSW; χSWR] *
        scheme.T[DNW DE DSW]
end

function _contract_edges_R(scheme::c3vCTM_honeycomb)
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.R[χNL DNW; χNR] *
        scheme.C[χNR χNE] * scheme.C[χNE; χSEL] * scheme.R[χSEL DE; χSER] *
        scheme.C[χSER; χS] * scheme.C[χS; χSWL] * scheme.R[χSWL DSW; χSWR] *
        conj(flip(scheme.T, [1 2 3])[DE DNW DSW])

end
