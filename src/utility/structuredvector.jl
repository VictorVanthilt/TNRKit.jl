struct StructuredVector{E, K, V, A <: AbstractVector{E}} <: AbstractVector{E}
    data::A
    structure::Dict{K, V}
end

"""
    StructuredVector(sv::SectorVector)

Construct a `StructuredVector` from a TensorKit `SectorVector`.
The flat data array and sector-index mapping are built automatically.

# Example
    StructuredVector(eig_vals(T))
"""
function StructuredVector end

# From a TensorKit SectorVector — preserves the charge sector organisation.
function StructuredVector(sv::TensorKit.SectorVector)
    structure = Dict(k => collect(r) for (k, r) in sv.structure)
    return StructuredVector(copy(sv.data), structure)
end

@inline Base.getindex(v::StructuredVector, i::Int) = getindex(parent(v), i)
@inline Base.getindex(v::StructuredVector{E, K}, keys::K) where {E, K} = parent(v)[v.structure[keys]]
@inline Base.setindex!(v::StructuredVector, val, i::Int) = setindex!(parent(v), val, i)

Base.size(v::StructuredVector, args...) = size(parent(v), args...)
Base.size(v::StructuredVector) = size(parent(v))
Base.copy(v::StructuredVector) = StructuredVector(copy(v.data), v.structure)
Base.parent(v::StructuredVector) = v.data

function Base.sort(v::StructuredVector; kwargs...)
    p = sortperm(v.data; kwargs...)
    inv_p = invperm(p)
    newdict = Dict(k => sort(inv_p[v.structure[k]]) for k in keys(v.structure))
    return StructuredVector(v.data[p], newdict)
end

function Base.filter(f, v::StructuredVector)
    kept_inds = findall(f, parent(v))
    data = parent(v)[kept_inds]
    old_to_new = Dict(old_ind => new_ind for (new_ind, old_ind) in enumerate(kept_inds))

    new_structure = Dict{keytype(v.structure), Vector{Int}}()
    for (sector, inds) in v.structure
        new_structure[sector] = [old_to_new[ind] for ind in inds if haskey(old_to_new, ind)]
    end

    return StructuredVector(data, new_structure)
end

function Base.show(io::IO, v::StructuredVector)
    println(io, "StructuredVector with keys: ", keys(v.structure))
    return print(io, "Data: ", v.data)
end

Base.:*(v::StructuredVector, x::Number) = StructuredVector(v.data .* x, v.structure)
Base.:*(x::Number, v::StructuredVector) = StructuredVector(x .* v.data, v.structure)
Base.:/(v::StructuredVector, x::Number) = StructuredVector(v.data ./ x, v.structure)
Base.:/(x::Number, v::StructuredVector) = StructuredVector(x ./ v.data, v.structure)

Base.keys(v::StructuredVector) = keys(v.structure)

# -- Broadcasting support -----------------------------------------------------
# Custom broadcast style so that element-wise operations preserve the
# StructuredVector container (and therefore the sector structure).
struct StructuredVectorStyle <: Broadcast.AbstractArrayStyle{1} end
StructuredVectorStyle(::Val{1}) = StructuredVectorStyle()  # parametric resize hook
Base.BroadcastStyle(::Type{<:StructuredVector}) = StructuredVectorStyle()
# Only scalars (and 0-dim arrays) get the StructuredVectorStyle result.
# Mixing with plain arrays is left undefined — it falls back to DefaultArrayStyle.
Base.BroadcastStyle(::StructuredVectorStyle, ::Broadcast.Style{Tuple}) = StructuredVectorStyle()
Base.BroadcastStyle(::Broadcast.Style{Tuple}, ::StructuredVectorStyle) = StructuredVectorStyle()
Base.BroadcastStyle(::StructuredVectorStyle, ::Broadcast.DefaultArrayStyle{0}) = StructuredVectorStyle()
Base.BroadcastStyle(::Broadcast.DefaultArrayStyle{0}, ::StructuredVectorStyle) = StructuredVectorStyle()

# Walk the broadcast tree to find the StructuredVector that determines the
# output structure.
_find_sv(bc::Broadcast.Broadcasted) = _find_sv(bc.args...)
_find_sv(sv::StructuredVector, rest...) = sv
_find_sv(::Any, rest...) = _find_sv(rest...)
_find_sv() = nothing

function Base.similar(bc::Broadcast.Broadcasted{StructuredVectorStyle}, ::Type{ElType}) where {ElType}
    sv = _find_sv(bc)
    if sv === nothing
        return similar(Array{ElType}, axes(bc))
    end
    return StructuredVector(similar(sv.data, ElType), copy(sv.structure))
end
