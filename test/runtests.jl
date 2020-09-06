using ReversibleSeismic
using Test

@testset "reversible" begin
    include("reversible.jl")
end

@testset "reversible_parallel" begin
    include("reversible_parallel.jl")
end

using CUDA
if CUDA.functional()
    @testset "cuda" begin
        include("cuda.jl")
    end

    @testset "reversible_gpu" begin
        include("reversible_gpu.jl")
    end
end