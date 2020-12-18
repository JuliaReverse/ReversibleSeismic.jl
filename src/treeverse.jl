function mid(δ, τ, σ, ϕ)
    ceil(Int, (δ*σ + τ*ϕ)/(τ+δ))
end

function treeverse!(step, s::T, state::Dict{Int,T}, δ, τ, β, σ, ϕ, depth=0) where T
    if σ > β
        δ -= 1
        state[β] = s
        for j=β:σ-1
            s = step(s)
        end
    end

    κ = mid(δ, τ, σ, ϕ)
    s_last = s
    counter = 0
    while τ>0 && κ < ϕ
        s_ = treeverse!(step, s, state, δ, τ, σ, κ, ϕ, depth+1)
        if counter == 0
            s_last = s_
        end
        counter += 1
        τ -= 1
        ϕ = κ
        κ = mid(δ, τ, σ, ϕ)
    end

    ϕ-σ > 1 && error("treeverse fails")
    #gs = gradient(step, s, gs)
    s = step(s)
    if σ>β
        pop!(state, β)
    end
    return s_last
end

function treeverse!(step, s::T, state::Dict{Int,T}; δ, τ) where T
    N = binomial(δ+τ, τ)
    treeverse!(step, s, state, δ, τ, 0, 0, N)
end

function directsolve(step, x0::T; nsteps::Int) where T
    x = copy(x0)
    for i=1:nsteps
        x = step(x)
    end
    return x
end

using Test
using Plots

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
    state = Dict{Int,Vector{FT}}()
    δ=3
    τ=2
    nsteps = binomial(τ+δ, τ)
    x_last = directsolve(step, FT.(x); nsteps=nsteps)
    s_last = treeverse!(step, FT.(x), state; τ=τ, δ=δ) |> step
    @show s_last |> sum
    plt = plot(s_last; label="treeverse")
    plot!(x_last; label="direct")
    display(plt)
    @test x_last ≈ s_last
end