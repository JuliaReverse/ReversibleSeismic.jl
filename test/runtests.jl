using ReversibleSeismic
using Test

@testset "reversible" begin
    include("reversible.jl")
end

@testset "reversible_parallel" begin
    include("reversible_parallel.jl")
end
