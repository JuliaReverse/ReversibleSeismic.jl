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

    function backward(y, x, g)
        _, gs = (~i_step_fun)(
            (GVar(y[1], g[1]), P3(GVar(y[2].x, g[2].x), GVar(y[2].y, g[2].y), GVar(y[2].z,g[2].z))),
            (GVar(x[1]), GVar(x[2])))
        NiLang.AD.grad(gs[1]), NiLang.AD.grad(gs[2])
    end

    @testset "treeverse gradient" begin
        x0 = P3(1.0, 0.0, 0.0)

        for N in [20, 120, 126]
            g_fd = ForwardDiff.gradient(x->rk4(lorentz, P3(x...), nothing; t0=0.0, Δt=3e-3, Nt=N)[end].x, [x0.x, x0.y, x0.z])
            g = (0.0, P3(1.0, 0.0, 0.0))
            g_tv, log = treeverse(step_fun, backward, (0.0, x0), g; δ=4, N=N)
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
        g, log = treeverse(step, ((a,b,c)->c+1), FT.(x), 0; N=nsteps, δ=δ)
        @test g == nsteps
        @test log.peak_mem[] == 3
        @test length(log.fcalls) == 2*nsteps+5
        @test length(log.gcalls) == nsteps
    end

    @testset "treeverse gradient" begin
        nx = ny = 100
        nstep = 2000
        c = 1000*ones(nx+2, ny+2)

        # gradient
        param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
            Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)
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
            tu = zeros(T, size(c)..., nstep+1)
            tφ = zeros(T, size(c)..., nstep+1)
            tψ = zeros(T, size(c)..., nstep+1)
            NiLang.AD.gradient(Val(1), i_loss!, (0.0, param, srci, srcj, srcv, c, tu, tφ, tψ))[end-3]
        end

        struct SeismicState{T}
            upre::Matrix{T}
            u::Matrix{T}
            φ::Matrix{T}
            ψ::Matrix{T}
            c::Matrix{T}
            step::Int
        end
        function SeismicState(c::Matrix{T}) where T
            SeismicState(zero(c), zero(c), zero(c), zero(c), c, 0)
        end
        Base.copy(s::SeismicState) = SeismicState(copy(s.upre), copy(s.u), copy(s.φ), copy(s.ψ), copy(s.c), s.step)
        Base.zero(s::SeismicState) = SeismicState(zero(s.upre), zero(s.u), zero(s.φ), zero(s.ψ), zero(s.c), 0)

        function step!(s)
            unext = zero(s.u)
            ReversibleSeismic.one_step!(param, unext, s.u, s.upre, s.φ, s.ψ, param.Σx, param.Σy, s.c)
            s2 = SeismicState(s.u, unext, s.φ, s.ψ, s.c, s.step+1)
            s2.u[srci, srcj] += srcv[s2.step]*param.DELTAT^2
            return s2
        end

        @i function i_step!(s2, s)
            @routine begin
                d2 ← zero(param.DELTAT)
                d2 += param.DELTAT^2
            end
            s2.upre .+= s.u
            s2.c .+= s.c
            ReversibleSeismic.i_one_step!(param, s2.u, s2.upre, s.upre,
                s2.φ, s.φ, s2.ψ, s.ψ, s2.c)
            s2.step += 1 + s.step
            s2.u[srci, srcj] += srcv[s2.step]* d2
            ~@routine
        end

        s1 = SeismicState(randn(102,102), randn(102,102), randn(102,102), randn(102,102), randn(102,102), 2)
        s3 = i_step!(zero(s1), copy(s1))[1]
        s4 = step!(copy(s1))
        @test s3.u[2:end-2,2:end-2] ≈ s4.u[2:end-2,2:end-2]
        @test s3.upre[2:end-2,2:end-2] ≈ s4.upre[2:end-2,2:end-2]
        @test s3.φ[2:end-2,2:end-2] ≈ s4.φ[2:end-2,2:end-2]
        @test s3.ψ[2:end-2,2:end-2] ≈ s4.ψ[2:end-2,2:end-2]
        @test s3.c ≈ s4.c
        @test s3.step ≈ s4.step

        function backward_step(y, x, g)
            gt = SeismicState([GVar(getfield(y, field), getfield(g, field)) for field in fieldnames(SeismicState)[1:end-1]]..., y.step)
            _, gs = (~i_step!)(gt, GVar(x))
            NiLang.AD.grad(gs)
        end
        g_nilang = getnilanggrad(c)
        state0 = SeismicState(copy(c))
        gn = SeismicState(zero(c))
        gn.u[45,45] = 1.0
        g_tv, log = treeverse(x->i_step!(zero(x), x)[1], backward_step, state0, gn; δ=20, N=nstep, f_inplace=true)
        @test isapprox(g_nilang, g_tv.c; rtol=1e-2, atol=1e-8)
        @show maximum(abs.(g_nilang - g_tv.c))
        @show maximum(abs.(g_nilang))
        @show maximum(abs.(g_tv.c))
    end
end
