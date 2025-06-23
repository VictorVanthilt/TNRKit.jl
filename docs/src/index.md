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
