using Test
using ReversibleSeismic, NiLang.AD, NiLang
using CUDA

@testset "sum instr" begin
    out = 0.4
    x = randn(5)
    CUDA.allowscalar(true)

    @test PlusEq(sum)(out, abs2, CuArray(x))[1] ≈ sum(abs2, x) + 0.4
    @test check_grad(PlusEq(sum), (out, abs2, x); iloss=1)
    @test check_grad(MinusEq(sum), (out, abs2, x); iloss=1)
    @test check_grad(PlusEq(sum), (out, abs2, x |> CuArray); iloss=1)
    @test check_grad(MinusEq(sum), (out, abs2, x |> CuArray); iloss=1)
    CUDA.allowscalar(false)
end


@testset "PlusEq(identity) instr - CUDA" begin
    x = randn(5) |> CuArray
    y = randn(5) |> CuArray
    @i function f(out, x, y)
        x .+= y
        out += sum((@skip! abs2), x)
    end
    out_, y_, x_ = (~f)(f(0.4, copy(y), copy(x))...)
    @test out_ ≈ 0.4
    @test x_ ≈ x
    @test y_ ≈ y
    CUDA.allowscalar(true)
    @test check_grad(f, (0.4, y, x); iloss=1)
    @test check_grad(~f, (0.4, y, x); iloss=1)

    @i function g(out, x, y)
        x[1:3] .+= y[1:3]
        out += sum((@skip! abs2), x)
    end
    out_, y_, x_ = (~g)(g(0.4, copy(y), copy(x))...)
    @test out_ ≈ 0.4
    @test x_ ≈ x
    @test y_ ≈ y
    @test check_grad(g, (0.4, y, x); iloss=1)
    CUDA.allowscalar(false)
end

