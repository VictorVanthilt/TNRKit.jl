function ind_pair(T::AbstractTensorMap, p::Tuple)
    p2 = filter(x -> !in(x, p), allind(T))
    return p, p2
end

# QR decomposition
function R1R2(A1, A2, p1, p2; check_space = true)
    p, q1 = ind_pair(A1, p1)
    _, RA1 = leftorth(A1, q1, p1;)
    p, q2 = ind_pair(A2, p2)
    RA2, _ = rightorth(A2, p2, q2)
    if check_space
        if domain(RA1) != codomain(RA2)
            @error "space mismatch"
        end
    end
    return RA1, RA2
end

# Find the pair of oblique projectors acting on the indices p1 of A1 and p2 of A2
#=
   ┌──┐        ┌──┐   
   │  ├◄──  ─◄─┤  │   
─◄─┤P1│        │P2├◄──
   │  ├◄──  ─◄─┤  │   
   └──┘        └──┘   
=#


function find_P1P2(A1, A2, p1, p2, trunc; check_space = true)
    R1, R2 = R1R2(A1, A2, p1, p2; check_space = check_space)
    return oblique_projector(R1, R2, trunc)
end

function oblique_projector(R1, R2, trunc;cutoff = 1e-16)
    mat = R1 * R2
    U, S, Vt = tsvd(mat; trunc = trunc & truncbelow(cutoff))

    P1 = R2 * adjoint(Vt) / sqrt(S)
    P2 = adjoint(U) * R1
    P2 = adjoint(adjoint(P2) / adjoint(sqrt(S)))
    return P1, P2
end

