# TNRKit

**Your one-stop-shop for Tensor Network Renormalization.**

# Package summary
TNRKit.jl aims to provide as many Tensor Network Renormalization methods as possible. Several models like the classical Ising, Potts and Six Vertex models are provided.

You can use TNRKit for calculating:
1. Partition functions (classical & quantum)
2. CFT data
3. Central charges

Many common TNR schemes have already been implemented:

**2D square tensor networks**
* [`TRG`](@ref) (Levin and Nave's Tensor Renormalization Group)
* [`BTRG`](@ref) (bond-weighted TRG)
* [`LoopTNR`](@ref) (entanglement filtering + loop optimization)
* [`SLoopTNR`](@ref) (c4 & inversion symmetric LoopTNR)
* [`HOTRG`](@ref) (higher order TRG)
* [`ATRG`](@ref) (anisotropic TRG)

**CTM methods (yet to be documented)**
* `ctm_TRG` (Corner Transfer Matrix environment + TRG)
* `ctm_HOTRG` (Corner Transfer Matrix environment + HOTRG)
* `c4CTM` (c4 symmetric CTM)
* `rCTM` (reflection symmetric CTM)

**3D cubic tensor networks**
* [`ATRG_3D`](@ref) (anisotropic TRG)
* [`HOTRG_3D`](@ref) (higher order TRG)

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
f = free_energy(data, ising_βc) # -2.1096504926141826902647832
```
You could even compare to the exact value, as calculated by the [Onsager solution](https://en.wikipedia.org/wiki/Ising_model#:~:text=Onsager%27s%20exact%20solution):

```julia-repl
julia> abs((f - f_onsager) / f_onsager)
3.1e-07
```
Pretty impressive for a calculation that takes about 0.3s on a laptop.

## Verbosity
There are 3 levels of verbosity implemented in TNRKit:
- Level 0: no TNRKit messages whatsoever.
- Level 1: Info at beginning and end of the simulations (including information on why the simulation stopped, how long it took and how many iterations were performed).
- Level 2: Level 1 + info at every iteration about the last generated finalize output and the iteration number.
  
to choose the verbosity level, simply use `run!(...; verbosity=n)`. The default is `verbosity=1`.

## Included Models
TNRKit includes several common models out of the box.
- Ising model: [`classical_ising`](@ref) and [`classical_ising_symmetric`](@ref), which has a Z2 grading on each leg.
- Potts model: [`classical_potts`](@ref) and [`classical_potts_symmetric`](@ref), which has a Zq grading on each leg.
- Six Vertex model: [`sixvertex`](@ref)
- Clock model: [`classical_clock`](@ref)
If you want to implement your own model you must respect the leg-convention assumed by all TNRKit schemes.

## Leg-convention
All the schemes assume that the input tensor lives in the space $V_1 \otimes V_2 \leftarrow V_3 \otimes V_4$ and that the legs are ordered in the following way:
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

The 3D scheme(s) assume that the input tensor lives in the space $V_{\text{D}} \otimes V_{\text{U}} \prime \leftarrow V_{\text{N}} \otimes V_{\text{E}} \otimes V_{\text{S}} \prime \otimes V_{\text{W}} \prime$.

Where D, U, N, E, S, W stand for Down, Up, North, East, South and West.