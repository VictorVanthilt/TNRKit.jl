abstract type stopcrit end

struct maxiter <: stopcrit
    n::Int
end

struct convcrit <: stopcrit
    Δ::Float64
    f::Function
end

struct MultipleCrit <: stopcrit
    crits::Vector{stopcrit}
end

Base.:&(a::stopcrit, b::stopcrit) = MultipleCrit([a, b])

(crit::maxiter)(steps::Int, data) = steps < crit.n
(crit::convcrit)(steps::Int, data) = crit.Δ < crit.f(steps, data)

# evaluate every criterion with short circuiting
(crit::MultipleCrit)(steps::Int, data) = !any(c -> !c(steps, data), crit.crits)
