using NiLang.AD
export treeverse, treeverse_solve

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
    treeverse!(f, gf, s, g; δ, N, τ=binomial_fit(N,δ), f_inplace=true)

Treeverse algorithm for back-propagating a program memory efficiently.

Positional arguments
* `f`, the step function that ``s_{i+1} = f(s_i)``,
* `gf`, the single step gradient function that ``g_i = gf(s_{i+1}, s_i, g_{i+1})``,
* `s`, the initial state ``s_0``,
* `g`, the gradient for the output state ``g_n``,

Keyword arguments
* `δ`, the number of checkpoints,
* `N`, the number of time steps,
* `τ`, the number of sweeps, it is chosen as the smallest integer that `binomial(τ+δ, τ) >= N` by default,
* `f_inplace = false`, whether `f` is inplace,
* `logger = TreeverseLog()`, the logger.

Ref: https://www.tandfonline.com/doi/abs/10.1080/10556789208805505
"""
function treeverse(f, gf, s::T, g; δ, N, τ=binomial_fit(N,δ), f_inplace=false, logger = TreeverseLog()) where T
    state = Dict{Int,typeof(s)}()
    if N > binomial(τ+δ, τ)
        error("please input a larger `τ` and `δ` so that `binomial(τ+δ, τ) >= N`!")
    end
    g = treeverse!(f, gf, s, state, g, δ, τ, 0, 0, N, logger, f_inplace)
    return g
end

function treeverse!(f, gf, s::T, state::Dict{Int,T}, g, δ, τ, β, σ, ϕ, logger, f_inplace) where T
    logger.depth[] += 1
    if σ > β
        δ -= 1
        # snapshot s
        state[β] = (f_inplace ? copy(s) : s)
        push!(logger.checkpoints, (τ, δ, logger.depth[], β))
        logger.peak_mem[] = max(logger.peak_mem[], length(state))
        for j=β:σ-1
            s = f(s)
            push!(logger.fcalls, (τ, δ, logger.depth[], j+1))
        end
    end

    κ = mid(δ, τ, σ, ϕ, δ)
    while τ>0 && κ < ϕ
        g = treeverse!(f, gf, f_inplace ? copy(s) : s, state, g, δ, τ, σ, κ, ϕ, logger, f_inplace)
        τ -= 1
        ϕ = κ
        κ = mid(δ, τ, σ, ϕ, δ)
    end

    if ϕ-σ != 1
        error("treeverse fails!")
    end
    q = f_inplace ? copy(s) : s
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

function treeverse_step!(s, param, srci, srcj, srcv, c)
    unext = zero(s.u)
    ReversibleSeismic.one_step!(param, unext, s.u, s.upre, s.φ, s.ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(s.u, unext, s.φ, s.ψ, s.step+1)
    s2.u[srci, srcj] += srcv[s2.step]*param.DELTAT^2
    return s2
end

function treeverse_grad(y, x, g, param, srci, srcj, srcv, gsrcv, c, gc)
    gt = SeismicState([GVar(getfield(y, field), getfield(g, field)) for field in fieldnames(SeismicState)[1:end-1]]..., y.step)
    _, gs, _, _, _, gv, gc2 = (~bennett_step!)(gt, GVar(x), param, srci, srcj, GVar(srcv, gsrcv), GVar(c, gc))
    (grad(gs), grad(gv), grad(gc2))
end

"""
    treeverse_solve(s0, gn; N, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
* `gn` is the gradient defined on the last state,
* `N` is the number of steps.
"""
function treeverse_solve(s0, gn; param, srci, srcj, srcv, c, N, δ=20, logger=TreeverseLog())
    treeverse(x->treeverse_step!(x, param, srci, srcj, srcv, c),
        (y,x,g)->treeverse_grad(y, x, g[1], param, srci, srcj, srcv, g[2], c, g[3]),
        copy(s0), gn; δ=δ, N=N, f_inplace=true, logger=logger)
end