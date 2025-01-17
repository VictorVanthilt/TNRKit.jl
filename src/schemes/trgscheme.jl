abstract type TRGScheme end

function step! end
function finalize! end

function run!(scheme::TRGScheme, trscheme::TensorKit.TruncationScheme, criterion::stopcrit;
              finalize_beginning=true, verbosity=1)
    data = []

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, scheme.finalize!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme)
            push!(data, scheme.finalize!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
        # @infov 1 "Elapsed time: $(t)s"
    end
    return data
end

function run!(scheme::TRGScheme, trscheme::TensorKit.TruncationScheme;
              finalize_beginning=true, verbosity=1)
    # default maxiter criterion of 100 iterations
    return run!(scheme, trscheme, maxiter(100); finalize_beginning=finalize_beginning,
                verbosity=verbosity)
end
