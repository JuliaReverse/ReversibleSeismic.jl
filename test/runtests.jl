using ReversibleSeismic
using Test, Pkg

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

if isinstalled("CUDA")
    @testset "cuda" begin
        include("cuda.jl")
    end
end

if isinstalled("KernelAbstractions")
    @testset "reversible_parallel" begin
        include("reversible_parallel.jl")
    end
end

if isinstalled("KernelAbstractions") && isinstalled("CUDA")
    @testset "reversible_gpu" begin
        include("reversible_gpu.jl")
    end
end
