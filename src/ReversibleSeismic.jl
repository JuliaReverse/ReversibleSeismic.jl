module ReversibleSeismic

using NiLang
using Requires

include("simulation.jl")
include("reversible.jl")
include("reversible_parallel.jl")
include("utils.jl")

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end
end
