using ReversibleSeismic, Test, ForwardDiff
using NiLang
using NiLang.AD: GVar
@testset "treeverse" begin
    struct P3{T}
        x::T
        y::T
        z::T
    end

    Base.zero(::Type{P3{T}}) where T = P3(zero(T), zero(T), zero(T))
    Base.zero(::P3{T}) where T = P3(zero(T), zero(T), zero(T))


    @inline function Base.:(+)(a::P3, b::P3)
        P3(a.x + b.x, a.y + b.y, a.z + b.z)
    end

    @inline function Base.:(/)(a::P3, b::Real)
        P3(a.x/b, a.y/b, a.z/b)
    end

    @inline function Base.:(*)(a::Real, b::P3)
        P3(a*b.x, a*b.y, a*b.z)
    end


    function lorentz(t, y, θ)
        P3(10*(y.y-y.x), y.x*(27-y.z)-y.y, y.x*y.y-8/3*y.z)
    end

    function rk4_step(f, t, y, θ; Δt)
        k1 = Δt * f(t, y, θ)
        k2 = Δt * f(t+Δt/2, y + k1 / 2, θ)
        k3 = Δt * f(t+Δt/2, y + k2 / 2, θ)
        k4 = Δt * f(t+Δt, y + k3, θ)
        return y + k1/6 + k2/3 + k3/3 + k4/6
    end

    function rk4(f, y0::T, θ; t0, Δt, Nt) where T
        history = zeros(T, Nt+1)
        history[1] = y0
        y = y0
        for i=1:Nt
            y = rk4_step(f, t0+(i-1)*Δt, y, θ; Δt=Δt)
            @inbounds history[i+1] = y
        end
        return history
    end


    @i @inline function :(+=)(identity)(Y::P3, X::P3)
        Y.x += X.x
        Y.y += X.y
        Y.z += X.z
    end

    @i @inline function :(+=)(*)(Y::P3, a::Real, X::P3)
        Y.x += a * X.x
        Y.y += a * X.y
        Y.z += a * X.z
    end

    @i @inline function :(+=)(/)(Y::P3, X::P3, b::Real)
        Y.x += X.x/b
        Y.y += X.y/b
        Y.z += X.z/b
    end

    @i function lorentz!(y!::P3{T}, t, y::P3{T}, θ) where T
        @routine @invcheckoff begin
            @zeros T a b c b_a ab αc ac
            a += y.x
            b += y.y
            c += y.z
            b_a += b-a
            ab += a * b
            αc += (8/3) * c
            c -= 27
            ac += a * c
        end
        y!.x += 10 * b_a
        y!.y -= ac + b
        y!.z += ab - αc
        ~@routine
    end

    @i function rk4_step!(f, y!::T, y::T, θ; Δt, t) where T
        @routine @invcheckoff begin
            @zeros T k1 k2 k3 k4 o1 o2 o3 o4 yk1 yk2 yk3
            f(o1, t, y, θ)
            k1 += Δt * o1
            yk1 += y
            yk1 += k1 / 2
            t += Δt/2
            f(o2, t, yk1, θ)
            k2 += Δt * o2
            yk2 += y
            yk2 += k2 / 2
            f(o3, t, yk2, θ)
            k3 += Δt * o3
            yk3 += y
            yk3 += k3
            t += Δt/2
            f(o4, t, yk3, θ)
            k4 += Δt * o4
        end
        y! += y
        y! += k1 / 6
        y! += k2 / 3
        y! += k3 / 3
        y! += k4 / 6
        ~@routine
    end

    @i function rk4!(f, history, y0::T, θ; t0, Δt, Nt) where T
        history[1] += y0
        @invcheckoff @inbounds for i=1:Nt
            rk4_step!(f, history[i+1], history[i], θ; Δt=Δt, t=t0+(i-1)*Δt)
        end
    end

    @i function iloss!(out, f, history, y0, θ; t0, Δt, Nt)
        rk4!((@const f), history, y0, θ; t0=t0, Δt=Δt, Nt=Nt)
        out += history[end].x
    end
    @i function i_step_fun(state2, state)
        rk4_step!((@const lorentz!), (state2 |> tget(2)), (state |> tget(2)), (); Δt=3e-3, t=state[1])
        (state2 |> tget(1)) += (state |> tget(1)) + 3e-3
    end

    function step_fun(x)
        i_step_fun((0.0, zero(x[2])), x)[1]
    end

    function backward(x, g)
        if g===nothing
            g = (0.0, P3(1.0, 0.0, 0.0))
        end
        y = step_fun(x)
        _, gs = (~i_step_fun)(
            (GVar(y[1], g[1]), P3(GVar(y[2].x, g[2].x), GVar(y[2].y, g[2].y), GVar(y[2].z,g[2].z))),
            (GVar(x[1]), GVar(x[2])))
        NiLang.AD.grad(gs[1]), NiLang.AD.grad(gs[2])
    end

    @testset "treeverse gradient" begin
        x0 = P3(1.0, 0.0, 0.0)
        for N in [20, 120, 126]
            g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
            g_tv = treeverse(step_fun, backward, (0.0, x0); δ=4, N=N)
            @test g_fd ≈ [g_tv[2].x, g_tv[2].y, g_tv[2].z]
        end
    end

    @testset "integrate" begin
        FT = Float64
        h = FT(0.01π)
        dt = FT(0.01)
        α = FT(1e-1)
        function step(src::AbstractArray{T}) where T
            dest = zero(src)
            n = length(dest)
            for i=1:n
                g = α*(src[mod1(i+1, n)] + src[mod1(i-1, n)] - 2*src[i]) / h^2
                dest[i] = src[i] + dt*g
            end
            return dest
        end
        n = 100
        x = zeros(FT, n)
        x[n÷2] = 1
        δ=3
        τ=2
        nsteps = binomial(τ+δ, τ)
        # directsolve
        log = ReversibleSeismic.TreeverseLog()
        g = treeverse(step, (b,c)-> c===nothing ? 1 : c+1, FT.(x); N=nsteps, δ=δ, logger=log)
        @test g == nsteps
        @test log.peak_mem[] == 4
        @test count(x->x.action==:call, log.actions) == 2*nsteps-5
        @test count(x->x.action==:grad, log.actions) == nsteps

        δ=3
        nsteps = 100000
        log = ReversibleSeismic.TreeverseLog()
        g = treeverse(x->x, (b,c)-> c===nothing ? 1 : c+1, 0.0; N=nsteps, δ=δ, logger=log)
        @test log.peak_mem[] == 4
        @test g == nsteps
        δ=9
        g = treeverse(x->x, (b,c)-> c===nothing ? 1 : c+1, 0.0; N=nsteps, δ=δ, logger=log)
        @test log.peak_mem[] == 10
        @test g == nsteps
    end

    @testset "treeverse gradient" begin
        nx = ny = 50
        N = 1000
        c = 1000*ones(nx+2, ny+2)

        # gradient
        param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
            Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=N)
        srci = size(c, 1) ÷ 2 - 1
        srcj = size(c, 2) ÷ 2 - 1
        srcv = Ricker(param, 100.0, 500.0)

        @i function i_loss!(out::T, param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T},
                tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
            i_solve!(param, srci, srcj, srcv, c, tu, tφ, tψ)
            out += tu[45,45,end]
        end

        function getnilanggrad(c::AbstractMatrix{T}) where T
            c = copy(c)
            tu = zeros(T, size(c)..., N+1)
            tφ = zeros(T, size(c)..., N+1)
            tψ = zeros(T, size(c)..., N+1)
            res = NiLang.AD.gradient(Val(1), i_loss!, (0.0, param, srci, srcj, srcv, c, tu, tφ, tψ))
            res[end-2], res[end-4], res[end-3]
        end

        s1 = ReversibleSeismic.SeismicState([randn(nx+2,ny+2) for i=1:4]..., 2)
        s3 = ReversibleSeismic.bennett_step!(zero(s1), copy(s1), param, srci, srcj, srcv, c)[1]
        s4 = ReversibleSeismic.treeverse_step(s1, param, srci, srcj, srcv, c)
        @test s3.u[2:end-2,2:end-2] ≈ s4.u[2:end-2,2:end-2]
        @test s3.upre[2:end-2,2:end-2] ≈ s4.upre[2:end-2,2:end-2]
        @test s3.φ[2:end-2,2:end-2] ≈ s4.φ[2:end-2,2:end-2]
        @test s3.ψ[2:end-2,2:end-2] ≈ s4.ψ[2:end-2,2:end-2]
        @test s3.step ≈ s4.step

        g_nilang_x, g_nilang_srcv, g_nilang_c = getnilanggrad(copy(c))
        s0 = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn = ReversibleSeismic.SeismicState(Float64, nx, ny)
        gn.u[45,45] = 1.0
        log = ReversibleSeismic.TreeverseLog()
        g_tv_x, g_tv_srcv, g_tv_c = treeverse_solve(s0, x->(gn, zero(srcv), zero(c));
                    param=param, c=copy(c), srci=srci, srcj=srcj,
                    srcv=srcv, δ=50, logger=log)
        @test isapprox(g_nilang_srcv, g_tv_srcv)
        @test isapprox(g_nilang_c, g_tv_c)
        @test maximum(g_nilang_c) ≈ maximum(g_tv_c)
        @test g_nilang_x[:,:,2] ≈ g_tv_x.u
    end
end
