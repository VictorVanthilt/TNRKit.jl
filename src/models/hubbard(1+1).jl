function R_tensor()
    R = zeros(ComplexF64, 2,2,2,2,2,2,2,2,2,2,2,2)
    for (pi11, pi12, pj11, pj12, pi21, pj22, i11, i12, j11, j12, i21, j22) in Iterators.product([0:1 for _ in 1:12]...)
        R[pi11+1, pi12+1, pj11+1, pj12+1, pi21+1, pj22+1, i11+1, i12+1, j11+1, j12+1, i21+1, j22+1] = 
            i11*i21 + i12*(i21 + pj11 + pi21 + pi11 + j11 + i22) + j11*(i21 + pj11 + pi21 + pi11) + 
            j12*(i21 + pj11 + pi21 + pi11 + i22 + pj12 + pi22 + pi12) + i22*(pj11 + pi21 + pi11) + pi22*(pj11 + pi21 + pi11 + pj12) +
            pi21*pj11 + pj12*(pj11 + pi11) + pi12*pi11
    end
    return R
end

