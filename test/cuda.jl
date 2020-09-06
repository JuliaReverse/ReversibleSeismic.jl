using Test
using ReversibleSeismic, CUDA, NiLang.AD

@testset "sum instr" begin
    out = 0.4
    x = randn(5)
    @test check_grad(PlusEq(sum), (out, abs2, x); iloss=1)
    @test check_grad(MinusEq(sum), (out, abs2, x); iloss=1)
    @test check_grad(PlusEq(sum), (out, abs2, x |> CuArray); iloss=1)
    @test check_grad(MinusEq(sum), (out, abs2, x |> CuArray); iloss=1)
end

