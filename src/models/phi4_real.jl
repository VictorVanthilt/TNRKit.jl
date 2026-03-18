#####################################
#       HELPER FUNCTIONS            #
#####################################

function f_real(ϕ1::Float64, ϕ2::Float64, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    return exp(
        -1 / 2 * (ϕ1 - ϕ2)^2
            - μ0 / 8 * (ϕ1^2 + ϕ2^2)
            - λ / 16 * (ϕ1^4 + ϕ2^4)
            + h / 4 * (ϕ1 + ϕ2)
    )
end

function fmatrix_real(ys::Vector{Float64}, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    K = length(ys)
    matrix = zeros(K, K)
    for i in 1:K
        for j in 1:K
            matrix[i, j] = f_real(ys[i], ys[j], μ0, λ, h)
        end
    end
    return TensorMap(matrix, ℂ^K ← ℂ^K)
end

function precompute_moments_real(K::Integer, μ0::Float64, λ::Float64)
    a = (4 + μ0) / 2
    b = λ / 4

    M = zeros(Float64, 4(K - 1) + 1)

    for n in 0:2:4(K - 1)   # only even n
        f(φ) = exp(-a * φ^2 - b * φ^4) * φ^n
        M[n + 1], _ = quadgk(f, -Inf, Inf)
    end
    return M
end


#####################################
#       TENSOR FUNCTIONS            #
#####################################

"""
    phi4_real(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    phi4_real(::Type{Z2Irrep}, K::Integer, μ0::Float64, λ::Float64)

Constructs the partition function tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation (and bond dimension) `K`, bare mass ``µ_0^2`` `μ0`, interaction constant `λ` and external field `h`.

Compatible with no symmetry or with explicit ℤ₂ symmetry on each of its spaces.
The latter is based on Taylor expanding the mixed sites term, and requires the external field `h` to be zero.
Defaults to ℤ₂ symmetry and `h = 0` if the symmetry type and magnetic field are not provided.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration. Has to be even for the ℤ₂ symmetric version.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `h::Float64`: External field (default is 0).

### Examples
```julia
    phi4_real(10, -1.0, 1.0, 1.0) # default ℤ₂ symmetry, h = 0
    phi4_real(Trivial, 10, -1.0, 1.0, 1.0) # no symmetry with magnetic field
```

!!! info
    When studying this model with impurities, the tensor without symmetry should be constructed, as the impurity breaks the ℤ₂ symmetry.

### References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)
* [Delcamp et. al. Phys. Rev. Research 2, 033278 (2020)](@cite delcamp2020)

See also: [`phi4_real_imp1`](@ref), [`phi4_real_imp2`](@ref).
"""
function phi4_real(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    return phi4_real(Z2Irrep, K, μ0, λ, h)
end
function phi4_real(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_real(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = svd_compact!(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                ws[p] * exp(ys[p]^2) *
                U[p, i] * U[p, j] * V[k, p] * V[l, p]
                for p in 1:K
            )
            for i in 1:K, j in 1:K, k in 1:K, l in 1:K
    ]

    T = TensorMap(T_arr, ℂ^K ⊗ ℂ^K ← ℂ^K ⊗ ℂ^K)
    return T
end
function phi4_real(::Type{Z2Irrep}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    if K % 2 != 0
        error("K must be even to split into even/odd groups")
    end

    logfact = log.(factorial.(0:(K - 1)))
    moments = precompute_moments_real(K, μ0, λ)

    T = zeros(Float64, K, K, K, K)

    perms = collect(permutations(1:4))  # 24 total

    # loop only over sorted tuples
    for s1 in 0:(K - 1), s2 in s1:(K - 1), s3 in s2:(K - 1), s4 in s3:(K - 1)
        n = s1 + s2 + s3 + s4
        if isodd(n)
            continue
        end

        M = moments[n + 1]
        denom_log = (logfact[s1 + 1] + logfact[s2 + 1] + logfact[s3 + 1] + logfact[s4 + 1]) / 2
        denom = exp(denom_log)

        val = M / denom

        # assign to all permutations
        idxs = (s1 + 1, s2 + 1, s3 + 1, s4 + 1)
        for p in perms
            ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
            T[ii, jj, kk, ll] = val
        end
    end

    # even/odd rearrangement
    evens = 1:2:K
    odds = 2:2:K
    perm = vcat(evens, odds)
    T = T[perm, perm, perm, perm]

    V = Z2Space(0 => K / 2, 1 => K / 2)
    return TensorMap(T, V ⊗ V ← V ⊗ V)
end


"""
    phi4_real_imp1([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)

Constructs the impurity tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation (and bond dimension) `K`, bare mass ``µ_0^2`` `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `h::Float64`: External field (default is 0).

### Examples
```julia
    phi4_real_imp1(10, -1.0, 1.0, 0.0)
```

### References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_real`](@ref), [`phi4_real_imp2`](@ref).
"""
function phi4_real_imp1(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    return phi4_real_imp1(Trivial, K, μ0, λ, h)
end
function phi4_real_imp1(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_real(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = svd_compact!(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                ys[p] * ws[p] * exp(ys[p]^2) *
                U[p, i] * U[p, j] * V[k, p] * V[l, p]
                for p in 1:K
            )
            for i in 1:K, j in 1:K, k in 1:K, l in 1:K
    ]

    T = TensorMap(T_arr, ℂ^K ⊗ ℂ^K ← ℂ^K ⊗ ℂ^K)
    return T
end


"""
    phi4_real_imp2([Type{Trivial}], K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)

Constructs the impurity tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation (and bond dimension) `K`, bare mass ``µ_0^2`` `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ^2 operator on this site.

It is based on [Gauss-Hermite quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature).

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Float64`: Bare mass. Note that in the calculation actually ``µ_0^2`` is used, but for readibility we write the ``µ_0^2`` as μ0
- `λ::Float64`: Coupling constant.
- `h::Float64`: External field (default is 0).

### Examples
```julia
    phi4_real_imp2(10, -1.0, 1.0, 0.0)
```

### References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadoh2019)

See also: [`phi4_real`](@ref), [`phi4_real_imp1`](@ref).
"""
function phi4_real_imp2(K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    return phi4_real_imp2(Trivial, K, μ0, λ, h)
end
function phi4_real_imp2(::Type{Trivial}, K::Integer, μ0::Float64, λ::Float64, h::Float64 = 0.0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_real(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = svd_compact!(f)

    # Make tensor for one site
    T_arr = [
        sum(
                √(S[i, i] * S[j, j] * S[k, k] * S[l, l]) *
                ys[p]^2 * ws[p] * exp(ys[p]^2) *
                U[p, i] * U[p, j] * V[k, p] * V[l, p]
                for p in 1:K
            )
            for i in 1:K, j in 1:K, k in 1:K, l in 1:K
    ]

    T = TensorMap(T_arr, ℂ^K ⊗ ℂ^K ← ℂ^K ⊗ ℂ^K)
    return T
end
