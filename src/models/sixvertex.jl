"""
    $(FUNCTIONNAME)(elt::Type{<:Number} = Float64, symmetry::Type{<:Sector} = Trivial; a = 1.0, b = 1.0, c = 1.0)
    $(FUNCTIONNAME)(symmetry::Type{<:Sector}; kwargs...)

Constructs the partition function tensor for the six-vertex model with a given symmetry type and coupling constants `a`, `b`, and `c`.

### Defaults
    - elt: Float64
    - symmetry: Trivial
    - a: 1.0
    - b: 1.0
    - c: 1.0


### Examples
```julia
    sixvertex() # Default symmetry is `Trivial`, coupling constants are `a = 1.0`, `b = 1.0`, `c = 1.0`.
    sixvertex(CU1Irrep) # CU1 symmetry with default coupling constants.
    sixvertex(ComplexF64, U1Irrep; a = 2.0, b = 3.0, c = 4.0) # U1 symmetry with custom coupling constants and element type.
```

"""
function sixvertex(
        elt::Type{<:Number} = ComplexF64, (::Type{Trivial}) = Trivial; a = 1.0, b = 1.0,
        c = 1.0
    )
    d = elt[
        a 0 0 0
        0 c b 0
        0 b c 0
        0 0 0 a
    ]
    return TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
end
function sixvertex(elt::Type{<:Number}, ::Type{U1Irrep}; a = 1.0, b = 1.0, c = 1.0)
    pspace = U1Space(-1 // 2 => 1, 1 // 2 => 1)
    mpo = zeros(elt, pspace ⊗ pspace, pspace ⊗ pspace)
    block(mpo, Irrep[U₁](0)) .= [b c; c b]
    block(mpo, Irrep[U₁](1)) .= reshape([a], (1, 1))
    block(mpo, Irrep[U₁](-1)) .= reshape([a], (1, 1))
    return mpo
end
function sixvertex(elt::Type{<:Number}, ::Type{CU1Irrep}; a = 1.0, b = 1.0, c = 1.0)
    pspace = CU1Space(1 // 2 => 1)
    mpo = zeros(elt, pspace ⊗ pspace, pspace ⊗ pspace)
    block(mpo, Irrep[CU₁](0, 0)) .= reshape([b + c], (1, 1))
    block(mpo, Irrep[CU₁](0, 1)) .= reshape([-b + c], (1, 1))
    block(mpo, Irrep[CU₁](1, 2)) .= reshape([a], (1, 1))
    return mpo
end
sixvertex(symmetry::Type{<:Sector}; kwargs...) = sixvertex(Float64, symmetry; kwargs...)
