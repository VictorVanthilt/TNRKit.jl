abstract type TNRScheme end

function step! end
function finalize! end

function run!(
        scheme::TNRScheme, trscheme::TensorKit.TruncationScheme, criterion::stopcrit;
        finalizer = default_Finalizer, finalize_beginning = true, verbosity = 1
    )

    data = output_type(finalizer)[]

    LoggingExtras.withlevel(; verbosity) do
        @infov 1 "Starting simulation\n $(scheme)\n"
        if finalize_beginning
            push!(data, finalizer.f!(scheme))
        end

        steps = 0
        crit = true

        t = @elapsed while crit
            @infov 2 "Step $(steps + 1), data[end]: $(!isempty(data) ? data[end] : "empty")"
            step!(scheme, trscheme)
            push!(data, finalizer.f!(scheme))
            steps += 1
            crit = criterion(steps, data)
        end

        @infov 1 "Simulation finished\n $(stopping_info(criterion, steps, data))\n Elapsed time: $(t)s\n Iterations: $steps"
    end
    return data
end
