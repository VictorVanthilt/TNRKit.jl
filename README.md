![TNRKit Logo](https://github.com/VictorVanthilt/TNRKit.jl/blob/master/assets/tensorlogo-dark.svg#gh-dark-mode-only)
![TNRKit Logo](https://github.com/VictorVanthilt/TNRKit.jl/blob/master/assets/tensorlogo.svg#gh-light-mode-only)

# TNRKit.jl

TNRKit.jl is a Julia package that aims to implement as much tensor network renormalization (TNR) schemes as possible.
It is built upon
[TensorKit.jl](https://github.com/jutho/TensorKit.jl), which provides functionality for
generic symmetries.
The following schemes are currently implemented:
- TRG (Levin and Nave's original formulation of a TNR scheme) 
- BTRG (bond-weighted TRG)
- ATRG (anisotropic TRG)
- HOTRG (higher order TRG)
- c4CTM (c4 symmetric CTM)

The project is not registered (yet) and is under active development. The interface is subject to changes. Any feedback about the user interface or the internals is much appreciated.

# Quick Start Guide
1. Choose a (TensorKit!) tensor that respects the leg-convention (see below)
2. Choose a TNR scheme
3. Choose a truncation scheme
4. Choose a stopping criterion

For example:
```julia
using TNRKit, TensorKit

T = classical_ising_symmetric(ising_βc) # partition function of classical Ising model at the critical point
scheme = BTRG(T) # Bond-weighted TRG (excellent choice)
data = run!(scheme, truncdim(16), maxiter(25)) # max bond-dimension of 16, for 25 iterations
```
`data` now contains 26 norms of the tensor, 1 for every time the tensor was normalized. (By default there is a normalization step before the first coarse-graining step wich can be turned off by changing the kwarg `run!(...; finalize_beginning=false)`)

Using these norms you could, for example, calculate the free energy of the critical classical Ising model:
```Julia
lnz = 0
for (i, d) in enumerate(data)
    lnz += log(d) * 2.0^(1 - i)
end

f_ising = lnz * -1 / ising_βc
```
You could even compare to the exact value, as calculated by Onsager:
```julia-repl
julia> abs((fs - f_onsager) / f_onsager)
3.1e-07
```
Pretty impressive for a calculation that takes about 0.3s on a modern laptop.
## Verbosity
There are 3 levels of verbosity implemented in TNRKit:
- Level 0: no TNRKit messages whatsoever.
- Level 1: Info at beginning and end of the simulations (including information on why the simulation stopped, how long it took and how many iterations were performed).
- Level 2: Level 1 + info at every iteration about the last generated finalize output and the iteration number.
  
to choose the verbosity level, simply use `run!(...; verbosity=n)`. The default is `verbosity=1`.

## Included Models
TNRKit includes several common models out of the box.
- Ising model: `classical_ising(β; h=0)` and `classical_ising_symmetric(β)`, which has a Z2 grading on each leg.
- Potts model: `classical_potts(q, β)`
- Six Vertex model: `sixvertex(scalartype, spacetype; a=1.0, b=1.0, c=1.0)`

If you want to implement your own model you must respect the leg-convention assumed by all TNRKit schemes.
## Leg-convention
All the schemes assume that the input tensor lives in the space `V₁⊗V₂←V₃⊗V₄` and that the legs are ordered in the following way:
```
     3
     |
     v
     |
1-<--┼--<-4
     |
     v
     |
     2
```
