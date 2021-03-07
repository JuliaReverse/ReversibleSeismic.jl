export treeverse

struct TreeverseLog
    fcalls::Vector{NTuple{4,Int}}  # τ, δ, function index f_i := s_{i-1} -> s_{i}, length should be `(2k-1)^n`
    gcalls::Vector{NTuple{4,Int}}  # τ, δ, function index
    checkpoints::Vector{NTuple{4,Int}}  # τ, δ, state index
    depth::Base.RefValue{Int}
    peak_mem::Base.RefValue{Int}  # should be `n*(k-1)+2`
end
TreeverseLog() = TreeverseLog(NTuple{4,Int}[], NTuple{4,Int}[], NTuple{4,Int}[], Ref(0), Ref(0))

function binomial_fit(N::Int, δ::Int)
    τ = 1
    while N > binomial(τ+δ, τ)
        τ += 1
    end
    return τ
end

function mid(δ, τ, σ, ϕ, d)
    κ = ceil(Int, (δ*σ + τ*ϕ)/(τ+δ))
    if κ >= ϕ && d > 0
        κ = max(σ+1, ϕ-1)
    end
    return κ
end

"""
    treeverse!(f, gf, s, g; δ, N, τ=binomial_fit(N,δ))

Treeverse algorithm for back-propagating a program memory efficiently.

Positional arguments
* `f`, the step function that ``s_{i+1} = f(s_i)``,
* `gf`, the single step gradient function that ``g_i = gf(s_{i+1}, s_i, g_{i+1})``,
* `s`, the initial state ``s_0``,
* `g`, the gradient for the output state ``g_n``,

Keyword arguments
* `δ`, the number of checkpoints,
* `N`, the number of time steps,
* `τ`, the number of sweeps, it is chosen as the smallest integer that `binomial(τ+δ, τ) >= N` by default.

Ref: https://www.tandfonline.com/doi/abs/10.1080/10556789208805505
"""
function treeverse(f, gf, s::T, g; δ, N, τ=binomial_fit(N,δ)) where T
    state = Dict{Int,typeof(s)}()
    if N > binomial(τ+δ, τ)
        error("please input a larger `τ` and `δ` so that `binomial(τ+δ, τ) >= N`!")
    end
    logger = TreeverseLog()
    g = treeverse!(f, gf, s, state, g, δ, τ, 0, 0, N, logger)
    return g, logger
end

function treeverse!(f, gf, s::T, state::Dict{Int,T}, g, δ, τ, β, σ, ϕ, logger) where T
    logger.depth[] += 1
    if σ > β
        δ -= 1
        # snapshot s
        state[β] = s
        push!(logger.checkpoints, (τ, δ, logger.depth[], β))
        logger.peak_mem[] = max(logger.peak_mem[], length(state))
        for j=β:σ-1
            s = f(s)
            push!(logger.fcalls, (τ, δ, logger.depth[], j+1))
        end
    end

    κ = mid(δ, τ, σ, ϕ, δ)
    while τ>0 && κ < ϕ
        g = treeverse!(f, gf, s, state, g, δ, τ, σ, κ, ϕ, logger)
        τ -= 1
        ϕ = κ
        κ = mid(δ, τ, σ, ϕ, δ)
    end

    if ϕ-σ != 1
        error("treeverse fails!")
    end
    q = s
    s = f(s)
    g = gf(s, q, g)
    push!(logger.fcalls, (τ, δ, logger.depth[], ϕ))
    push!(logger.gcalls, (τ, δ, logger.depth[], ϕ))
    if σ>β
        # retrieve s
        s = pop!(state, β)
    end
    return g
end