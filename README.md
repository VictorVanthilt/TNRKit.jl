# TNRKit.jl
---
Is a Julia package that aims to implement as much tensor network renormalization (TNR) schemes as possible.
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
