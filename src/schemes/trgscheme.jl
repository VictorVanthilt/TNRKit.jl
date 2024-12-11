abstract type TRGScheme end

function step! end
function finalize! end

function run!(scheme::TRGScheme, trscheme::TensorKit.TruncationScheme, criterion::stopcrit; finalize_beginning=true)

    data = []
    if finalize_beginning
        @info "Finalizing beginning"
        push!(data, scheme.finalize!(scheme))
    end

    steps = 0    
    crit = true
    
    while crit
        @info "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
        step!(scheme, trscheme)
        push!(data, scheme.finalize!(scheme))
        steps += 1
        crit = criterion(steps, data)
    end
    return data
end

function run!(scheme::TRGScheme, trscheme::TensorKit.TruncationScheme; finalize_beginning=true)
    # default maxiter criterion of 100 iterations
    return run!(scheme, trscheme, maxiter(100), finalize_beginning=finalize_beginning)
end