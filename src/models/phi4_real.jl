#####################################
#       HELPER FUNCTIONS            #
#####################################

function f_real(ϕ1, ϕ2, μ0, λ, h = 0)
    return exp(
        -1 / 2 * (ϕ1 - ϕ2)^2
            - μ0 / 8 * (ϕ1^2 + ϕ2^2)
            - λ / 16 * (ϕ1^4 + ϕ2^4)
            + h / 4 * (ϕ1 + ϕ2)
    )
end

function fmatrix_real(ys, μ0, λ, h = 0)
    K = length(ys)
    matrix = zeros(K, K)
    for i in 1:K
        for j in 1:K
            matrix[i, j] = f_real(ys[i], ys[j], μ0, λ, h)
        end
    end
    return TensorMap(matrix, ℂ^K ← ℂ^K)
end

function precompute_moments_real(K, μ0, λ)
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
$(SIGNATURES)

Constructs the partition function tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

### Arguments
- `K::Int`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.
- `h::Number`: External field (default is 0).

### Examples
```julia
    phi4_real(10, -1, 1, 0)
```

### References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadohTensorNetworkAnalysis2019)

See also: [`phi4_real_imp1`](@ref), [`phi4_real_imp2`](@ref), [`phi4_real_symmetric`](@ref).
"""
function phi4_real(K::Integer, μ0::Number, λ::Number, h::Number = 0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_real(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = tsvd(f)

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


"""
$(SIGNATURES)

Constructs the impurity tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ operator on this site.

### Arguments
- `K::Int`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.
- `h::Number`: External field (default is 0).

### Examples
```julia
    phi4_real_imp1(10, -1, 1, 0)
```

### References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadohTensorNetworkAnalysis2019)

See also: [`phi4_real`](@ref), [`phi4_real_imp2`](@ref), [`phi4_real_symmetric`](@ref).
"""
function phi4_real_imp1(K::Integer, μ0::Number, λ::Number, h::Number = 0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_real(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = tsvd(f)

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
$(SIGNATURES)

Constructs the impurity tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ^2 operator on this site.

### Arguments
- `K::Int`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.
- `h::Number`: External field (default is 0).

### Examples
```julia
    phi4_real_imp2(10, -1, 1, 0)
```

### References
* [Kadoh et. al. 10.1007/JHEP05(2019)184 (2019)](@cite kadohTensorNetworkAnalysis2019)

See also: [`phi4_real`](@ref), [`phi4_real_imp1`](@ref), [`phi4_real_symmetric`](@ref).
"""
function phi4_real_imp2(K::Integer, μ0::Number, λ::Number, h::Number = 0)
    # Weights and locations
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_real(ys, μ0, λ, h)

    # SVD fmatrix
    U, S, V = tsvd(f)

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


"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D square lattice
for the real ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

This tensor has explicit ℤ₂ symmetry on each of its spaces.

### Arguments
- `K::Int`: Number of quadrature points for Gauss-Hermite integration. K has to be even!
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

!!! info
    `h` is always 0.

### Examples
```julia
    phi4_real_symmetric(10, -1, 1)
```

### References
* [Delcamp et. al. Phys. Rev. Research 2, 033278 (2020)](@cite delcampComputingRenormalizationGroup2020)

See also: [`phi4_real`](@ref), [`phi4_real_imp1`](@ref), [`phi4_real_symmetric`](@ref).
"""
function phi4_real_symmetric(K::Integer, μ0::Number, λ::Number)
    if K % 2 != 0
        error("K must be even to split into even/odd groups")
    end

    logfact = log.(factorial.(0:(K - 1)))
    moments = precompute_moments_real(K, μ0, λ)

    T = zeros(Float64, K, K, K, K)

    perms = collect(permutations(1:4))  # 24 total

    # loop only over sorted tuples
    for s1 in 0:(K - 1)
        for s2 in s1:(K - 1)
            for s3 in s2:(K - 1)
                for s4 in s3:(K - 1)

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
            end
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
