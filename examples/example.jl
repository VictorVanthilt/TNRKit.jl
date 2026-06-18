using Revise, TensorKit, TNRKit

# choose a TensorKit truncation scheme
trunc = truncrank(16) & trunctol(atol = 1.0e-40)

# ---- TRG with the new iterable interface ----

# create a pure algorithm config (kwargs with sensible defaults)
alg = TRG(; trunc = trunc, maxiter = 25)
renorm = Renormalizer(alg, classical_ising(1.0))

# iterate manually to inspect intermediate states
for (state, norms) in renorm
    τ0, _ = extract_tau_and_c(state.T; fast = true)
    # compute observables at each step...
end

# extract results after iteration
T_final = get_tensor(renorm)
f = free_energy(renorm.norms, 1.0)

# ---- BTRG ----

# criterion to determine convergence
trg_f(steps::Int, data) = abs(log(data[end]) * 2.0^(-steps))
stopping_criterion = convcrit(1.0e-16, trg_f) & maxiter(20)

# initialize and run the BTRG scheme
scheme = BTRG(classical_ising(1.0), -0.5)
data = run!(scheme, trunc, stopping_criterion)

# ---- HOTRG ----

# initialize and run the HOTRG scheme
scheme = HOTRG(classical_ising(1.0))
data = run!(scheme, trunc, stopping_criterion)
