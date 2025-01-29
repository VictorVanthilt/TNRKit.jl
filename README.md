# TNRKit.jl
---
TNRKit.jl is a Julia package that aims to implement as much tensor network renormalization (TNR) schemes as possible.
It is built upon
[TensorKit.jl](https://github.com/jutho/TensorKit.jl), which provides functionality for
generic symmetries.
The following schemes are currently implemented:
- TRG (Levin and Nave's original formulation of a TNR scheme) 
- BTRG (bond-weighted TRG)
- ATRG (anisotropic TRG)
- HOTRG (higher order TRG)
- GILTTNR (graph independent local truncation + TRG, this still needs some improvements)
- SLoopTNR (experimental symmetric Loop TNR)

The project is not registered (yet) and is under active development. The interface is subject to changes. Any feedback about the user interface or the internals is much appreciated.

# Quick Start Guide
1. Choose a (TensorKit!) tensor that respects the leg-convention (see below)
2. Choose a TNR scheme
3. Choose a truncation scheme
4. Choose a stopping criterion

For example:
```julia
T = classical_ising_symmetric(Ising_βc) # partition function of classical Ising model at the critial point
scheme = BTRG(T) # Bond-weighted TRG (excellent choice)
data = run!(scheme, truncdim(16), maxiter(25)) # max bond-dimension of 16, for 25 iterations
```
`data` now contains 26 norms of the tensor, 1 for every time the tensor was normalized. (By default there is a normalization step at the very beginning wich can be turned off by setting the kwarg `finalize_beginning=false` in `run!`)

Using these norms you could, for example, calculate the free energy of the critial classical Ising model:
```Julia
lnz = 0
for (i, d) in enumerate(data)
    lnz += log(d) * 2.0^(1 - i)
end

f_ising = lnz * -1 / Ising_βc
```
You could even compare to the exact value, as calculated by Onsager:
```julia-repl
julia> abs((fs - f_onsager) / f_onsager)
3.1e-07
```
Pretty impressive for a calculation that takes about 0.3s on a modern laptop.

### Leg-convention
All the schemes assume that the input tensor lives in the space `V₁⊗V₂←V₃⊗V₄` and that the legs are ordered in the following way:
```
     4
     |
     v
     |
1-<--┼--<-3
     |
     v
     |
     2
```
