using ReversibleSeismic, CUDA
using Test, Pkg

@testset "simulation" begin
    include("simulation.jl")
end

@testset "reversible" begin
    include("reversible.jl")
end

@testset "bennett" begin
    include("bennett.jl")
end

@testset "treeverse" begin
    include("treeverse.jl")
end

function isinstalled(target)
    deps = Pkg.dependencies()
    for (uuid, dep) in deps
        dep.is_direct_dep || continue
        dep.name == target && return true
    end
    return false
end

if CUDA.functional()
    @testset "cuda" begin
        include("cuda.jl")
    end
    @testset "reversible_gpu" begin
        include("reversible_gpu.jl")
    end
end
