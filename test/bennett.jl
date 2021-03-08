using NiLang
using ReversibleSeismic
using Test

@testset "integrate" begin
    k = 20
    nx = ny = 100
    N = 1999

    upre = zeros(nx+2, ny+2)
    u = zeros(nx+2, ny+2)
    φ = zeros(nx+2, ny+2)
    ψ = zeros(nx+2, ny+2)

    param = AcousticPropagatorParams(nx=nx, ny=ny,
        Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=N)
    c = 1000*ones(nx+2, ny+2)
    srci = nx ÷ 2
    srcj = ny ÷ 2
    srcv = Ricker(param, 100.0, 500.0)

    x = ReversibleSeismic.SeismicState(upre, u, φ, ψ, 0)

    x_last = NiLang.direct_emulate(bennett_step!, copy(x), param, srci, srcj, srcv, copy(c); N=N)
    x_last_b = bennett(bennett_step!, zero(x), copy(x), param, srci, srcj, srcv, copy(c); k=k, N=N)[2]
    @test isapprox(x_last.u, x_last_b.u; atol=1e-5)
    @test sum(abs2, x_last.u) ≈ 6.234873084873294e-5
    @test isapprox(x_last.ψ, x_last_b.ψ; atol=1e-5)
    @test isapprox(x_last.φ, x_last_b.φ; atol=1e-5)
    @test isapprox(x_last.upre, x_last_b.upre; atol=1e-5)
    @test x_last.step == x_last_b.step

    @i function loss(out, step, y, x, param, srci, srcj, srcv, c; kwargs...)
        bennett((@skip! step), y, x, param, srci, srcj, srcv, c; kwargs...)
        out += y.u[srci, srcj]
    end
    _,_,_,gx,_,_,_,_,gc = NiLang.AD.gradient(loss, (0.0, bennett_step!, zero(x), copy(x), param, srci, srcj, srcv, copy(c)); iloss=1, k=k, N=N)
    x_last_2 = NiLang.direct_emulate(bennett_step!, (x2=copy(x); x2.u[srci, srcj]+=1e-5; x2), param, srci, srcj, srcv, copy(c); N=N)
    @test gx.u[srci, srcj] ≈ (x_last_2.u[srci, srcj] - x_last.u[srci, srcj])/1e-5
    x_last_3 = NiLang.direct_emulate(bennett_step!, copy(x), param, srci, srcj, srcv, (c2=copy(c); c2[srci, srcj]+=1e-5; c2); N=N)
    @test isapprox(gc[srci, srcj], (x_last_3.u[srci, srcj] - x_last.u[srci, srcj])/1e-5; rtol=1e-3)

    gn = ReversibleSeismic.SeismicState(Float64, nx, ny)
    gn.u[srci, srcj] = 1
    gx2, gv, gc2 = bennett_solve(x, gn; param=param, srci=srci, srcj=srcj, srcv=srcv, c=c, N=N, k=k)
    @test gx.u ≈ gx2.u
    @test gc ≈ gc2
end
