#####################################
#       HELPER FUNCTIONS            #
#####################################

function f_complex(ℝϕ1, ℂϕ1, ℝϕ2, ℂϕ2, μ0, λ)
    return exp(
        -1 / 2 * ((ℝϕ1 - ℝϕ2)^2 + (ℂϕ1 - ℂϕ2)^2)
            - μ0 / 8 * (ℝϕ1^2 + ℂϕ1^2 + ℝϕ2^2 + ℂϕ2^2)
            - λ / 16 * ((ℝϕ1^2 + ℂϕ1^2)^2 + (ℝϕ2^2 + ℂϕ2^2)^2)
    )
end


function fmatrix_complex(ys, μ0, λ)
    K = length(ys)
    matrix = zeros(K^2, K^2)
    @threads for i in 1:K
        for j in i:K
            for k in 1:K
                for l in 1:K
                    idx1 = (i - 1) * K + j
                    idx2 = (k - 1) * K + l
                    if idx2 >= idx1  # only compute upper triangle
                        val = f_complex(ys[i], ys[j], ys[k], ys[l], μ0, λ)
                        matrix[idx1, idx2] = val
                        matrix[idx2, idx1] = val  # symmetric counterpart

                        # Based on the simultaneous symmetry of (i,j)<->(j,i) and (k,l)<->(l,k)
                        idx3 = (j - 1) * K + i
                        idx4 = (l - 1) * K + k
                        matrix[idx3, idx4] = val
                        matrix[idx4, idx3] = val  # symmetric counterpart
                    end
                end
            end
        end
    end
    return TensorMap(matrix, ℂ^(K^2) ← ℂ^(K^2))
end


function precompute_moments_complex(K, μ0, λ)
    a = (4 + μ0) / 2
    b = λ / 4
    nmax = 8 * (K - 1) + 1
    M = zeros(Float64, nmax + 1)

    for n in 0:nmax
        f(φ) = begin
            logval = n * log(φ) - a * φ^2 - b * φ^4
            return exp(logval)        # safe everywhere, never NaN
        end

        val, _ = quadgk(f, 0.0, Inf; rtol = 1.0e-8, maxevals = 10^7)
        M[n + 1] = val * 2π
    end
    return M
end


#####################################
#       TENSOR FUNCTIONS            #
#####################################

"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

### Examples
```julia
    phi4_complex(10, -1., 1.)
```

See also: [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref), [`phi4_complex_symmetric`](@ref).
"""
function phi4_complex(K::Integer, μ0::Number, λ::Number)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    N = K^2
    T_arr = zeros(eltype(S), N, N, N, N)

    weights = [ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
                    for α in 1:K, β in 1:K
                        s += factor *
                            weights[α, β] *
                            U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] *
                            V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i, j, k, l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii, jj, kk, ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end

"""
$(SIGNATURES)

Constructs the impurity tensor for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ operator on this site.
    
### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕ(10, -1., 1.)
```

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref), [`phi4_complex_symmetric`](@ref).
"""
function phi4_complex_impϕ(K::Integer, μ0::Number, λ::Number)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [(ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
                    for α in 1:K, β in 1:K
                        s += factor *
                            weights[α, β] *
                            U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] *
                            V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i, j, k, l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii, jj, kk, ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


"""
$(SIGNATURES)

Constructs the impurity tensor for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕ† operator on this site.
    
### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕdag(10, -1., 1.)
```

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref), [`phi4_complex_symmetric`](@ref).
"""
function phi4_complex_impϕdag(K::Integer, μ0::Number, λ::Number)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [(ys[α] - ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
                    for α in 1:K, β in 1:K
                        s += factor *
                            weights[α, β] *
                            U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] *
                            V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i, j, k, l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii, jj, kk, ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end

"""
$(SIGNATURES)

Constructs the impurity tensor for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

The impurity is a √(ϕϕ†) operator on this site.
    
### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕabs(10, -1., 1.)
```

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref), [`phi4_complex_symmetric`](@ref).
"""
function phi4_complex_impϕabs(K::Integer, μ0::Number, λ::Number)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
                    for α in 1:K, β in 1:K
                        s += factor *
                            weights[α, β] *
                            U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] *
                            V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i, j, k, l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii, jj, kk, ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end

"""
$(SIGNATURES)

Constructs the impurity tensor for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

The impurity is a ϕϕ† operator on this site.
    
### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

### Examples
```julia
    phi4_complex_impϕ2(10, -1., 1.)
```

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_all`](@ref), [`phi4_complex_symmetric`](@ref).
"""
function phi4_complex_impϕ2(K::Integer, μ0::Number, λ::Number)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    N = K^2
    T_arr = zeros(ComplexF64, N, N, N, N)

    weights = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]

    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
                    for α in 1:K, β in 1:K
                        s += factor *
                            weights[α, β] *
                            U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] *
                            V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i, j, k, l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii, jj, kk, ll] = s
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T
end


"""
$(SIGNATURES)

Constructs all the tensors: the partition function tensor and all the impurity tensors for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

It is faster to compute them all at once then one for one individually.
    
### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.

### Examples
```julia
    phi4_complex_all(10, -1., 1.)
```

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_symmetric`](@ref).
"""
function phi4_complex_all(K::Integer, μ0::Number, λ::Number)
    ys, ws = gausshermite(K)

    # Determine fmatrix
    f = fmatrix_complex(ys, μ0, λ)

    # SVD fmatrix
    U, S, V = tsvd(f)

    N = K^2


    T_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕ_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕdag_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕabs_arr = zeros(ComplexF64, N, N, N, N)
    T_ϕ2_arr = zeros(ComplexF64, N, N, N, N)

    w = [ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕ = [(ys[α] + ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕdag = [(ys[α] - ys[β]im) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕabs = [sqrt(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]
    w_ϕ2 = [(ys[α]^2 + ys[β]^2) * ws[α] * ws[β] * exp(ys[α]^2 + ys[β]^2) for α in 1:K, β in 1:K]


    perms = collect(permutations(1:4))  # 24 total

    @threads for i in 1:N
        for j in i:N
            for k in j:N
                for l in k:N
                    s = 0.0
                    s_ϕ = 0.0
                    s_ϕdag = 0.0
                    s_ϕabs = 0.0
                    s_ϕ2 = 0.0
                    factor = √(S[i, i] * S[j, j] * S[k, k] * S[l, l])
                    for α in 1:K, β in 1:K
                        s += factor * w[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                        s_ϕ += factor * w_ϕ[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                        s_ϕdag += factor * w_ϕdag[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                        s_ϕabs += factor * w_ϕabs[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                        s_ϕ2 += factor * w_ϕ2[α, β] * U[(α - 1) * K + β, i] * U[(α - 1) * K + β, j] * V[k, (α - 1) * K + β] * V[l, (α - 1) * K + β]
                    end

                    # Fill all 24 symmetric permutations
                    idxs = (i, j, k, l)
                    for p in perms
                        ii, jj, kk, ll = idxs[p[1]], idxs[p[2]], idxs[p[3]], idxs[p[4]]
                        T_arr[ii, jj, kk, ll] = s
                        T_ϕ_arr[ii, jj, kk, ll] = s_ϕ
                        T_ϕdag_arr[ii, jj, kk, ll] = s_ϕdag
                        T_ϕabs_arr[ii, jj, kk, ll] = s_ϕabs
                        T_ϕ2_arr[ii, jj, kk, ll] = s_ϕ2
                    end
                end
            end
        end
    end

    T = TensorMap(T_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕ = TensorMap(T_ϕ_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕdag = TensorMap(T_ϕdag_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕabs = TensorMap(T_ϕabs_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    T_ϕ2 = TensorMap(T_ϕ2_arr, ℂ^N ⊗ ℂ^N ← ℂ^N ⊗ ℂ^N)
    return T, T_ϕ, T_ϕdag, T_ϕabs, T_ϕ2
end


"""
$(SIGNATURES)

Constructs the partition function tensor for a 2D square lattice
for the complex ϕ^4 model with a given approximation `K`, bare mass µ_0^2 `μ0`, interaction constant `λ` and external field `h`.

This tensor has explicit U(1) symmetry on each of its spaces.
    
### Arguments
- `K::Integer`: Number of quadrature points for Gauss-Hermite integration.
- `μ0::Number`: Bare mass. Note that in the calculation actually µ_0^2 is used, but for readibility we write the µ_0^2 as μ0
- `λ::Number`: Coupling constant.
- `μ::Number`: Chemical potential in the y direction. (Default μ=0, so no chemical potential)

### Examples
```julia
    phi4_complex_symmetric(10, -1., 1.; μ=0)
```

See also: [`phi4_complex`](@ref), [`phi4_complex_impϕ`](@ref), [`phi4_complex_impϕdag`](@ref), [`phi4_complex_impϕabs`](@ref), [`phi4_complex_impϕ2`](@ref), [`phi4_complex_all`](@ref).
"""
function phi4_complex_symmetric(K::Integer, μ0::Number, λ::Number; μ::Number = 0)
    if K % 2 != 0
        error("K must be even to split into even/odd groups")
    end

    # precompute
    moments = precompute_moments_complex(K, μ0, λ)
    # log factorials 0..K-1
    logfact = log.(factorial.(0:(K - 1)))
    # precompute exp(μ*s/2) for s = 0 .. 2*(K-1)
    maxsum = 2 * (K - 1)
    E = exp.((μ / 2) .* (0:maxsum))   # E[s+1] = exp( μ*s/2 )

    T_arr = zeros(Float64, K, K, K, K, K, K, K, K)

    @threads for r1 in 0:(K - 1)
        for r2 in 0:(K - 1)
            for r3 in 0:(K - 1)
                for r4 in 0:(K - 1)
                    # precompute sums depending only on r's
                    rsum = r1 + r2 + r3 + r4
                    r24 = r2 + r4     # used in exp factor
                    # clever computational trick
                    logr = logfact[r1 + 1] + logfact[r2 + 1] + logfact[r3 + 1] + logfact[r4 + 1]

                    for l1 in 0:(K - 1)
                        for l2 in 0:(K - 1)
                            for l3 in 0:(K - 1)
                                # solve delta for l4:
                                # r1 + r2 + l3 + l4 = r3 + r4 + l1 + l2
                                l4 = r3 + r4 + l1 + l2 - r1 - r2 - l3

                                if l4 < 0 || l4 > K - 1
                                    continue
                                end

                                # total power n
                                n = rsum + 1 + l1 + l2 + l3 + l4
                                # quick skip if moment is zero
                                M = moments[n + 1]
                                if M == 0.0
                                    continue
                                end

                                # exp factor: exp( μ*(r2+r4 - l2 - l4)/2 ) =
                                #   E[r24]/E[l2+l4]
                                l24 = l2 + l4
                                expfactor = E[r24 + 1] / E[l24 + 1]

                                # denomenator via logfacts
                                logdenom = 0.5 * (
                                    logr + logfact[l1 + 1] + logfact[l2 + 1] +
                                        logfact[l3 + 1] + logfact[l4 + 1]
                                )
                                denom = exp(logdenom)

                                val = M * expfactor / denom

                                # store into array (indices +1)
                                T_arr[r1 + 1, l1 + 1, r2 + 1, l2 + 1, r3 + 1, l3 + 1, r4 + 1, l4 + 1] = val
                            end
                        end
                    end
                end
            end
        end
    end

    # Build U1 spaces
    V1 = U1Space([U1Irrep(q) => 1 for q in 0:(K - 1)]...)
    V2 = U1Space([U1Irrep(q) => 1 for q in 0:-1:(-K + 1)]...)
    T_unfused = TensorMap(T_arr, V1 ⊗ V2 ⊗ V1 ⊗ V2 ← V1 ⊗ V2 ⊗ V1 ⊗ V2)

    U = isometry(fuse(V1, V2), V1 ⊗ V2)
    Udg = adjoint(U)

    @tensor T_fused[-1 -2; -3 -4] := T_unfused[1 2 3 4; 5 6 7 8] * U[-1; 1 2] * U[-2; 3 4] * Udg[5 6; -3] * Udg[7 8; -4]
    return T_fused
end
