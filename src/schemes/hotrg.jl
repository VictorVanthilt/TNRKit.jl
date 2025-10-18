"""
$(TYPEDEF)

Higher-Order Tensor Renormalization Group

### Constructors
    $(FUNCTIONNAME)(T [, finalize=finalize!])

### Running the algorithm
    run!(::HOTRG, trunc::TensorKit.TruncationSheme, stop::Stopcrit[, finalize_beginning=true, verbosity=1])

Each step rescales the lattice by a (linear) factor of 2

!!! info "verbosity levels"
    - 0: No output
    - 1: Print information at start and end of the algorithm
    - 2: Print information at each step

### Fields

$(TYPEDFIELDS)

### References
* [Xie et. al. Phys. Rev. B 86 (2012)](@cite xieCoarsegrainingRenormalizationHigherorder2012)

"""
mutable struct HOTRG <: TNRScheme
    T::TensorMap

    finalize!::Function
    function HOTRG(T::TensorMap{E, S, 2, 2}; finalize = (finalize!)) where {E, S}
        @assert [isdual(space(T, ax)) for ax in 1:4] == [0, 0, 1, 1]
        return new(T, finalize)
    end
end

#= 
NOTE: It's difficult to transpose the PFTensors planarly 
in order to reuse the y-compression code for x-compression.
Hence both are written explicitly.
=#

function _get_hotrg_xproj(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    #= join in y-direction, keep x-indices open (A1 below A2)
    M M†                        M† M
            ┌---1---┐                   ┌---1---┐
            ↓       ↑                   ↑       ↓
    -1 -←--A2-←-2--A2†-←- -3    -1 -←--A2†--2-←-A2-←- -3
            ↓       ↑                   ↑       ↓
            5       6                   5       6
            ↓       ↑                   ↑       ↓
    -2 -←--A1-←-4--A1†-←- -4    -2 -←--A1†--4-←-A1-←- -4
            ↓       ↑                   ↑       ↓
            └---3---┘                   └---3---┘
    =#
    @plansor MM[-1 -2; -3 -4] :=
        A2[-1 5; 1 2] * A1[-2 3; 5 4] *
        conj(A2[-3 6; 1 2]) * conj(A1[-4 3; 6 4])
    sL, L = eigh!(MM)
    L = L * pseudopow(sL, 0.5)
    @plansor MM[-1 -2; -3 -4] :=
        conj(A2[2 5; 1 -1]) * conj(A1[4 3; 5 -2]) *
        A2[2 6; 1 -3] * A1[4 3; 6 -4]
    sR, R = eigh!(MM)
    R = pseudopow(sR, 0.5) * R'
    # construct projectors
    U, s, Vh, ε = tsvd!(R * L; trunc)
    sinv_sqrt = pseudopow(s, -0.5)
    P1 = L * Vh' * sinv_sqrt
    P2 = sinv_sqrt * U' * R
    return P1, P2, ε
end

function _get_hotrg_yproj(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    #= join in x-direction, keep y-indices open (A1 on the left of A2)
    M M†                        M† M
            -3      -4              -3      -4
            ↓       ↓               ↓       ↓
        ┌-→-A1†--6-→A2†-→┐      ┌-←-A1-←-6--A2-←-┐
        ↑   ↓       ↓    ↓      ↑   ↓       ↓    ↓
        1   2       4    3      1   2       4    3
        ↑   ↓       ↓    ↓      ↑   ↓       ↓    ↓
        └-←-A1-←-5--A2-←-┘      └-→-A1†--5-→A2†-→┘
            ↓       ↓               ↓       ↓
            -1      -2              -1      -2
    =#
    # get bottom unitary
    @plansor MM[-1 -2; -3 -4] :=
        A1[1 -1; 2 5] * A2[5 -2; 4 3] *
        conj(A1[1 -3; 2 6]) * conj(A2[6 -4; 4 3])
    sL, L = eigh!(MM)
    L = L * pseudopow(sL, 0.5)
    # get top unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A1[1 2; -1 5]) * conj(A2[5 4; -2 3]) *
        A1[1 2; -3 6] * A2[6 4; -4 3]
    sR, R = eigh!(MM)
    R = pseudopow(sR, 0.5) * R'
    # construct projectors
    U, s, Vh, ε = tsvd!(R * L; trunc)
    sinv_sqrt = pseudopow(s, -0.5)
    P1 = L * Vh' * sinv_sqrt
    P2 = sinv_sqrt * U' * R
    return P1, P2, ε
end

function _step_hotrg_y(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        P1::AbstractTensorMap{E, S, 2, 1}, P2::AbstractTensorMap{E, S, 1, 2}
    ) where {E, S}
    #= compression along the y-direction
                    -3
                    |
            ┌---1---A2---3--┐
            |       |       |
        -1--P2      5      P1-- -4
            |       |       |
            └---2---A1---4--┘
                    |
                    -2
    =#
    @tensor T[-1 -2; -3 -4] :=
        P2[-1; 1 2] * P1[3 4; -4] * A2[1 5; -3 3] * A1[2 -2; 5 4]
    return T
end

function _step_hotrg_x(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        P1::AbstractTensorMap{E, S, 2, 1}, P2::AbstractTensorMap{E, S, 1, 2}
    ) where {E, S}
    #= compression along the x-direction
                -3
                |
            ┌3--P1-4┐
            |       |
        -1--A1--5---A2-- -4
            |       |
            └1--P2-2┘
                |
                -2
    =#
    @tensor T[-1 -2; -3 -4] :=
        A1[-1 1; 3 5] * A2[5 2; 4 -4] * P2[-2; 1 2] * P1[3 4; -3]
    return T
end

function step!(scheme::HOTRG, trunc::TensorKit.TruncationScheme)
    P1, P2, = _get_hotrg_xproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_y(scheme.T, scheme.T, P1, P2)
    P1, P2, = _get_hotrg_yproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_x(scheme.T, scheme.T, P1, P2)
    return scheme
end

function Base.show(io::IO, scheme::HOTRG)
    println(io, "HOTRG - Higher Order TRG")
    println(io, "  * T: $(summary(scheme.T))")
    return nothing
end
