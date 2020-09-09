using ReversibleSeismic
using Test, Pkg

@testset "reversible" begin
    include("simulation.jl")
end

@testset "reversible" begin
    include("reversible.jl")
end

function isinstalled(target)
    deps = Pkg.dependencies()
    for (uuid, dep) in deps
        dep.is_direct_dep || continue
        dep.name == target && return true
    end
    return false
end

if isinstalled("KernelAbstractions")
    @testset "cuda" begin
        include("cuda.jl")
    end
    @testset "reversible_parallel" begin
        include("reversible_parallel.jl")
    end
    @testset "reversible_gpu" begin
        include("reversible_gpu.jl")
    end
end
