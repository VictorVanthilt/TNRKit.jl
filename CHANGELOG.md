# Changelog

## TNRKit v0.5.0

[Diff since v0.4.0](https://github.com/VictorVanthilt/TNRKit.jl/compare/v0.4.0...v0.5.0)

TNRKit v0.5.0 adds new features and significantly unifies the model interface.

## Breaking Changes

### Unified Model Interface (#149)

All models that previously had a separate `functionname_symmetric` variant for symmetry-enhanced tensors
have been unified into a single function that takes the symmetry type as the first positional argument.
The default behavior of each function now returns a tensor with the **maximum available symmetry**.

| Old (v0.4.0) | New (v0.5.0) |
|---|---|
| `classical_ising(β)` | `classical_ising(Trivial, β)` |
| `classical_ising_symmetric(β)` | `classical_ising(β)` or `classical_ising(Z2Irrep, β)` |
| `classical_ising_3D(β)` | `classical_ising_3D(Trivial, β)` |
| `classical_ising_symmetric_3D(β)` | `classical_ising_3D(β)` or `classical_ising_3D(Z2Irrep, β)` |
| `classical_ising_triangular(β)` | `classical_ising_triangular(Trivial, β)` |
| `classical_ising_triangular_symmetric(β)` | `classical_ising_triangular(β)` or `classical_ising_triangular(Z2Irrep, β)` |
| `classical_potts(q, β)` | `classical_potts(Trivial, q, β)` |
| `classical_potts_symmetric(q, β)` | `classical_potts(q, β)` or `classical_potts(ZNIrrep{q}, q, β)` |
| `classical_clock(q, β)` | `classical_clock(Trivial, q, β)` |
| `classical_clock_symmetric(q, β)` | `classical_clock(q, β)` or `classical_clock(ZNIrrep{q}, q, β)` |
| `classical_XY_U1_symmetric(β, n)` | `classical_XY(β, n)` or `classical_XY(U1Irrep, β, n)` |
| `classical_XY_O2_symmetric(β, n)` | `classical_XY(CU1Irrep, β, n)` |
| `phi4_real(K, μ0, λ)` | `phi4_real(Trivial, K, μ0, λ)` |
| `phi4_real_symmetric(K, μ0, λ)` | `phi4_real(K, μ0, λ)` or `phi4_real(Z2Irrep, K, μ0, λ)` |
| `phi4_complex(K, μ0, λ)` | `phi4_complex(Trivial, K, μ0, λ)` |
| `phi4_complex_symmetric(K, μ0, λ)` | `phi4_complex(K, μ0, λ)` or `phi4_complex(U1Irrep, K, μ0, λ)` |

The following exported names have been **removed**:
- `classical_ising_symmetric`
- `classical_ising_symmetric_3D`
- `classical_ising_triangular_symmetric`
- `classical_potts_symmetric`
- `classical_clock_symmetric`
- `classical_XY_U1_symmetric`
- `classical_XY_O2_symmetric`
- `phi4_real_symmetric`
- `phi4_complex_symmetric`

All model functions now also accept a `T::Type{<:Number}` keyword argument to control the element type of the output tensor.

### LoopTNR `run!` Interface (#144)

The six-argument `run!` call for `LoopTNR` that took separate truncation and stopping criterion arguments for the entanglement optimization has been replaced by the `LoopParameters` struct, which bundles all loop-optimization parameters:

```julia
# Old (v0.4.0)
run!(scheme, trscheme, truncentanglement, criterion, entanglement_criterion, loop_criterion)

# New (v0.5.0)
run!(scheme, trscheme, criterion, LoopParameters())
```

The simplified two-argument form `run!(scheme, trscheme, criterion)` still works and uses `LoopParameters()` defaults internally.

## New Methods

### Correlation Functions
- **`CorrelationHOTRG`**: Computes two-point correlation functions using HOTRG. Calculates correlators between two impurities at a horizontal distance of 2ⁿ sites (#139).

### Honeycomb Lattice CTM
- **`c3vCTM_honeycomb`**: C3v-symmetric Corner Transfer Matrix for honeycomb lattices (#141).

## New Models

### Honeycomb Lattice Ising Model (#141)
- `classical_ising_honeycomb`: Ising model on the honeycomb lattice
- New constants: `ising_βc_honeycomb` and `f_onsager_honeycomb`

### XY Model
- `classical_XY`: Unified XY model function (replaces `classical_XY_U1_symmetric` and `classical_XY_O2_symmetric`)
- New constant: `XY_βc`

## Added Functionality

### Nuclear Norm Regularized LoopTNR (#144)
- `LoopParameters`: New exported struct to configure loop optimization, including support for the nuclear norm regularization (NNR-TNR).
- `VN_entropy`: Calculate the von Neumann entropy of a transfer matrix.
- `loop_entropy`: Calculate the circular and radial entanglement entropies of a `LoopTNR` scheme.

## Bugfixes
- Fix wrong twist factors and switch from `svd_trunc` to `eigh_trunc` for projectors in 3D HOTRG (#145).
- Fix broken onesite CTM implementation (#155).
- Fix incorrect symmetric complex φ⁴ tensor construction, now accessible via `phi4_complex(U1Irrep, ...)` (#142).
- Remove incorrect mentions of external field `h` from complex φ⁴ docstrings and ensure all tensors use `ComplexF64` (#147).

**Merged pull requests:**
- CorrelationHOTRG (#139) (@JaridPiceu)
- Implement c3vCTM on the honeycomb lattice (#141) (@sanderdemeyer)
- Fix complex phi4 (#142) (@JaridPiceu)
- Update README with new phi4 models (#143) (@JaridPiceu)
- Nuclear norm regularized LoopTNR (#144) (@Chenqitrg)
- Fix twist and use eigh_trunc in 3D HOTRG (#145) (@Yue-Zhengyuan)
- Add Code Coverage (#146) (@VictorVanthilt)
- Small Improvement Complex phi4 model (#147) (@JaridPiceu)
- Small improvements to the LoopTNR docstrings (#148) (@Chenqitrg)
- [Breaking] Redefine functions to take symmetries as argument (#149) (@borisdevos)
- LoopTNR docstring typos (#150) (@Chenqitrg)
- Added the reference and test of classical XY model (#152) (@Chenqitrg)
- Fix onesite CTM (#155) (@VictorVanthilt)
- Adding two references about CFT data calculation (#156) (@Yue-Zhengyuan)
- Test Krylov and NNR LoopTNR methods (#157) (@VictorVanthilt)
