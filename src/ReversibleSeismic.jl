module ReversibleSeismic

using NiLang
using Requires

include("simulation.jl")
include("reversible.jl")
include("utils.jl")
include("bennett.jl")
include("treeverse.jl")
include("neuralode.jl")
include("detector.jl")

function __init__()
    @require KernelAbstractions="63c18a36-062a-441e-b654-da1e3ab1ce7c" begin
        @require CUDAKernels = "72cfdca4-0801-4ab0-bf6a-d52aa10adc57" begin
            include("cuda.jl")
            include("reversible_parallel.jl")
        end
    end
end

end
