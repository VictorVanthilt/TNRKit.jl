mutable struct VBTRG{E, S} <: TNRScheme
    T::TensorMap{E, S, 2, 2}
    finalize!::Function
    function VBTRG(T::TensorMap{E, S, 2, 2}; finalize = (finalize!)) where {E, S}
        @assert sectortype(T) == Trivial "Only trivial sector type is supported for the tensor T"
        return new{scalartype(T), spacetype(T)}(T, finalize)
    end
end

# Vertical projector code
mutable struct _vbtrg_vertical_projector{E, S}
    mps::MPSKit.InfiniteMPS
    w::TensorMap{E, S, 1, 2}
    function _vbtrg_vertical_projector(mps::MPSKit.InfiniteMPS, χv::Int)
        w = randn(scalartype(mps), spacetype(mps)(χv) ← space(mps.AL[1], 2) ⊗ space(mps.AC[1], 2))
        return new{scalartype(w), spacetype(w)}(mps, w)
    end
end

function trΩ(x::_vbtrg_vertical_projector)
    return norm(x.mps)
end

function trΩ_w_dagw(x::_vbtrg_vertical_projector)
    return @tensoropt x.mps.AL[1][7 1; 2] * x.mps.AC[1][2 3; 9] * x.w[8; 1 3] * conj(x.w[8; 4 5]) * conj(x.mps.AL[1][7 4; 6]) * conj(x.mps.AC[1][6 5; 9])
end

function Γ(x::_vbtrg_vertical_projector)
    return @tensoropt Γ[-2 -3; -1] := x.mps.AL[1][4 -2; 6] * x.mps.AC[1][6 -3; 5] * conj(x.w[-1; 2 3]) * conj(x.mps.AL[1][4 2; 1]) * conj(x.mps.AC[1][1 3; 5])
end

# Horizontal projector code
mutable struct _vbtrg_horizontal_projector{E, S}
    mps::MPSKit.InfiniteMPS
    env::MPSKit.InfiniteEnvironments
    w::TensorMap{E, S, 2, 1}
    function _vbtrg_horizontal_projector(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, χh::Int)
        w = randn(scalartype(mps), space(env.GLs[1], 2) ⊗ space(env.GLs[1], 2) ← spacetype(mps)(χh))
        return new{scalartype(w), spacetype(w)}(mps, env, w)
    end
end

function trΩ(x::_vbtrg_horizontal_projector)
    return @tensoropt x.env.GLs[1][9 4; 1] * x.mps.C[1][1; 3] * x.env.GRs[1][3 4; 8] * conj(x.env.GLs[1][9 7; 5]) * conj(x.mps.C[1][5; 6]) * conj(x.env.GRs[1][6 7; 8])
end

function trΩ_w_dagw(x::_vbtrg_horizontal_projector)
    return @tensoropt x.env.GLs[1][1 2; 7] * x.mps.C[1][7; 10] * x.env.GRs[1][10 5; 4] * x.w[2 3; 11] * conj(x.env.GLs[1][1 3; 9]) * conj(x.mps.C[1][9; 8]) * conj(x.env.GRs[1][8 6; 4]) * conj(x.w[5 6; 11])
end

function Γ(x::_vbtrg_horizontal_projector)
    return @tensoropt Γ[-3; -1 -2] := x.env.GLs[1][6 -1; 8] * x.mps.C[1][8; 4] * x.env.GRs[1][4 3; 1] * conj(x.w[3 2; -3]) * conj(x.env.GLs[1][6 -2; 7]) * conj(x.mps.C[1][7; 5]) * conj(x.env.GRs[1][5 2; 1])
end

# Intermediate projector code
mutable struct _vbtrg_intermediate_projector_trbl{E, S}
    mps::MPSKit.InfiniteMPS
    env::MPSKit.InfiniteEnvironments
    Ttr::TensorMap{E, S, 1, 2}
    Tbl::TensorMap{E, S, 2, 1}
    w::TensorMap{E, S, 1, 1}
    function _vbtrg_intermediate_projector_trbl(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, T::TensorMap{E, S, 2, 2}, χq::Int) where {E, S}
        Tbl, Ttr = SVD12(T, truncdim(χq))
        w = randn(scalartype(mps), spacetype(mps)(χq) ← space(Ttr, 1))
        return new{scalartype(w), spacetype(w)}(mps, env, Ttr, Tbl, w)
    end
end

function trΩ(x::_vbtrg_intermediate_projector_trbl)
    return @tensoropt x.env.GLs[1][1 3; 7] * x.mps.AC[1][7 5; 4] * x.env.GRs[1][4 6; 9] * x.Ttr[8; 5 6] * x.Tbl[3 2; 8] * conj(x.mps.AC[1][1 2; 9])
end

function trΩ_w_dagw(x::_vbtrg_intermediate_projector_trbl)
    return @tensoropt x.env.GLs[1][2 1; 9] * x.mps.AC[1][9 5; 7] * x.env.GRs[1][7 6; 11] * x.Ttr[ 8; 5 6] * x.w[10; 8] * conj(x.w[10; 4]) * x.Tbl[1 3; 4] * conj(x.mps.AC[1][2 3; 11])
end

function Γ(x::_vbtrg_intermediate_projector_trbl)
    return @tensoropt Γ[-1; -2] := x.env.GLs[1][2 1; 8] * x.mps.AC[1][8 7; 5] * x.Ttr[-1; 7 6] * x.env.GRs[1][5 6; 9] * conj(x.mps.AC[1][2 3; 9]) * x.Tbl[1 3; 4] * conj(x.w[-2; 4])
end

mutable struct _vbtrg_intermediate_projector_tlbr{E, S}
    mps::MPSKit.InfiniteMPS
    env::MPSKit.InfiniteEnvironments
    Ttl::TensorMap{E, S, 1, 2}
    Tbr::TensorMap{E, S, 2, 1}
    w::TensorMap{E, S, 1, 1}
    function _vbtrg_intermediate_projector_tlbr(mps::MPSKit.InfiniteMPS, env::MPSKit.InfiniteEnvironments, T::TensorMap{E, S, 2, 2}, χq::Int) where {E, S}
        Ttl, Tbr = SVD12(permute(T, ((1, 3), (2, 4))), truncdim(χq))
        Ttl = permute(Ttl, ((1,), (2, 3)))
        Tbr = permute(Tbr, ((1, 2), (3,)))
        w = randn(scalartype(mps), space(Ttl, 3) ← spacetype(mps)(χq))
        return new{scalartype(w), spacetype(w)}(mps, env, Ttl, Tbr, w)
    end
end

function trΩ(x::_vbtrg_intermediate_projector_tlbr) # Done
    return @tensoropt x.env.GLs[1][1 2; 3] * x.mps.AC[1][3 4; 5] * x.env.GRs[1][5 8; 6] * x.Ttl[2; 4 9] * x.Tbr[9 7; 8] * conj(x.mps.AC[1][1 7; 6])
end

function trΩ_w_dagw(x::_vbtrg_intermediate_projector_tlbr) # Done
    return @tensoropt x.env.GLs[1][1 2; 3] * x.mps.AC[1][3 4; 5] * x.env.GRs[1][5 8; 6] * x.Ttl[2; 4 11] * x.w[11; 10] * conj(x.w[9; 10]) * x.Tbr[9 7; 8] * conj(x.mps.AC[1][1 7; 6])
end

function Γ(x::_vbtrg_intermediate_projector_tlbr)
    return @tensoropt Γ[-1; -2] := x.env.GLs[1][1 2; 3] * x.mps.AC[1][3 4; 5] * x.Ttl[2; 4 -2] * x.env.GRs[1][5 6; 7] * conj(x.mps.AC[1][1 8; 7]) * x.Tbr[9 8; 6] * conj(x.w[9; -1])
end

function run!(scheme::VBTRG, trunc::TensorKit.TruncationDimension, criterion::stopcrit; inner_trunc = trunc, VUMPSalg = VUMPS(), envdim::Int = trunc.dim)
    data = []
    iter = 0
    crit = true

    push!(data, scheme.finalize!(scheme))

    # 1. Initialize VUMPS mps and env
    mps = InfiniteMPS(randn, Float64, space(scheme.T, 1), spacetype(scheme.T)(envdim))
    mpo = MPO(PeriodicArray([scheme.T]))
    mps, env, _ = leading_boundary(mps, mpo, VUMPSalg)

    while crit
        # 2. Optimize each of the 3 projectors
        vert = _vbtrg_vertical_projector(mps, trunc.dim)
        optimize!(vert, trunc; convcrit = 1.0e-9)

        horiz = _vbtrg_horizontal_projector(mps, env, trunc.dim)
        optimize!(horiz, trunc; convcrit = 1.0e-9)

        qstruct1 = _vbtrg_intermediate_projector_trbl(mps, env, scheme.T, inner_trunc.dim)
        optimize!(qstruct1, inner_trunc; convcrit = 1.0e-9)

        qstruct2 = _vbtrg_intermediate_projector_tlbr(mps, env, scheme.T, inner_trunc.dim)
        optimize!(qstruct2, inner_trunc; convcrit = 1.0e-9)

        # 3. Update the tensor and the environment
        mps = _update_boundary(mps, vert)
        env = _update_environment(env, horiz)
        _update_tensor!(scheme, vert, horiz, qstruct1, qstruct2)

        # 4. Run VUMPS again -> loop to step 2
        mps, env, _ = leading_boundary(mps, InfiniteMPO(PeriodicVector([scheme.T])), VUMPS(; maxiter = 1), env)

        iter += 1

        push!(data, scheme.finalize!(scheme))
        crit = criterion(iter, data)
    end
    return data
end

const VBTRG_PROJECTOR = Union{_vbtrg_vertical_projector, _vbtrg_horizontal_projector, _vbtrg_intermediate_projector_trbl, _vbtrg_intermediate_projector_tlbr}

function _update_projector!(proj::VBTRG_PROJECTOR, trunc::TensorKit.TruncationScheme)
    U, S, Vdag = tsvd(Γ(proj); trunc = trunc)
    proj.w = adjoint(Vdag) * adjoint(U)

    # rescale
    C1 = trΩ(proj)
    C2 = trΩ_w_dagw(proj)
    proj.w = proj.w * sqrt(C1 / C2)
    return S
end

function optimize!(proj::VBTRG_PROJECTOR, trunc::TensorKit.TruncationScheme; maxiter::Int = 100, convcrit::Float64 = 1.0e-6)
    S = _update_projector!(proj, trunc)

    for iter in 2:maxiter
        Scurr = _update_projector!(proj, trunc)

        if space(S) == space(Scurr)
            normdiff = norm(S - Scurr)
            if normdiff <= convcrit
                @info "optimize! converged in $iter iterations | normdiff = $normdiff"
                return proj
            end
        end
        S = Scurr
    end
    @warn "optimize! did not converge in $maxiter iterations"
    return proj
end

function _update_boundary(mps::MPSKit.InfiniteMPS, proj::_vbtrg_vertical_projector)
    return InfiniteMPS([@tensor AL′[-1, -2; -3] := mps.AL[1][-1; 1 2] * mps.AL[1][2 3; -3] * proj.w[-2; 1 3]])
end

function _update_environment(env::MPSKit.InfiniteEnvironments, proj::_vbtrg_horizontal_projector)
    @tensoropt GL′[-1 -2; -3] := env.GLs[1][3 2; -3] * proj.w[2 1; -2] * conj(env.GLs[1][3 1; -1])
    @tensoropt GR′[-1 -2; -3] := env.GRs[1][-1 2; 1] * conj(proj.w[2 3; -2]) * conj(env.GRs[1][-3 3; 1])
    return MPSKit.InfiniteEnvironments(PeriodicVector([GL′]), PeriodicVector([GR′]))
end

function _update_tensor!(scheme::VBTRG, vert::_vbtrg_vertical_projector, horiz::_vbtrg_horizontal_projector, qstruct1::_vbtrg_intermediate_projector_trbl, qstruct2::_vbtrg_intermediate_projector_tlbr)

    # Make North block
    @tensoropt N[-1; -2 -3] := qstruct1.Ttr[4; 1 2] * qstruct2.Ttl[2; 3 5] * qstruct1.w[-1; 4] * qstruct2.w[5; -3] * conj(vert.w[-2; 1 3])

    # Make East block
    @tensoropt E[-1 -2; -3] := qstruct1.w[-1; 4] * qstruct1.Ttr[4; 2 3] * qstruct2.Tbr[5 2; 1] * horiz.w[1 3; -3] * conj(qstruct2.w[5; -2])

    # Make South block
    @tensoropt S[-1 -2; -3] := conj(qstruct2.w[1; -2]) * qstruct2.Tbr[1 2; 3] * qstruct1.Tbl[3 4; 5] * vert.w[-1; 2 4] * conj(qstruct1.w[-3; 5])

    # Make West block
    @tensoropt W[-1; -2 -3] := conj(qstruct1.w[-2; 1]) * qstruct1.Tbl[2 3; 1] * qstruct2.Ttl[4 3; 5] * qstruct2.w[5; -3] * conj(horiz.w[2 4; -1])

    # Combine blocks
    @tensoropt T[-1 -2; -3 -4] := N[1; -3 2] * E[3 2; -4] * S[-2 4; 3] * W[-1; 1 4]
    scheme.T = T
    return scheme
end
