using NiLang.AD
export treeverse, treeverse_solve

struct TreeverseAction
    action::Symbol
    τ::Int
    δ::Int
    step::Int
    depth::Int
end

struct TreeverseLog
    actions::Vector{TreeverseAction}
    depth::Base.RefValue{Int}
    peak_mem::Base.RefValue{Int}  # should be `n*(k-1)+2`
end
TreeverseLog() = TreeverseLog(TreeverseAction[], Ref(0), Ref(0))
Base.push!(tlog::TreeverseLog, args...) = push!(tlog.actions, TreeverseAction(args..., tlog.depth[]))

Base.show(io::IO, ::MIME"text/plain", logger::TreeverseLog) = Base.show(io, logger)
function Base.show(io::IO, logger::TreeverseLog)
    print(io, """Treeverse log
| peak memory usage = $(logger.peak_mem[])
| number of function calls = $(count(x->x.action==:call, logger.actions))
| number of gradient calls = $(count(x->x.action==:grad, logger.actions))
| number of stack push/pop = $(count(x->x.action==:store, logger.actions))/$(count(x->x.action==:fetch, logger.actions))""")
end

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
    treeverse(f, gf, s, g; δ, N, τ=binomial_fit(N,δ), f_inplace=true, logger = TreeverseLog())

Treeverse algorithm for back-propagating a program memory efficiently.

Positional arguments
* `f`, the step function that ``s_{i+1} = f(s_i)``,
* `gf`, the single step gradient function that ``g_i = gf(s_i, g_{i+1})``,
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
        push!(logger, :store, τ, δ, β)
        logger.peak_mem[] = max(logger.peak_mem[], length(state))
        for j=β:σ-1
            s = f(s)
            push!(logger, :call, τ, δ, j)
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
    g = gf(s, g)
    push!(logger, :grad, τ, δ, σ)
    if σ>β
        # retrieve s
        pop!(state, β)
        push!(logger, :fetch, τ, δ, β)
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

function treeverse_grad(x, g, param, srci, srcj, srcv, gsrcv, c, gc)
    y = treeverse_step!(copy(x), param, srci, srcj, srcv, c)
    gt = SeismicState([GVar(getfield(y, field), getfield(g, field)) for field in fieldnames(SeismicState)[1:end-1]]..., y.step)
    _, gs, _, _, _, gv, gc2 = (~bennett_step!)(gt, GVar(x), param, srci, srcj, GVar(srcv, gsrcv), GVar(c, gc))
    (grad(gs), grad(gv), grad(gc2))
end

"""
    treeverse_solve(s0, gn; param, srci, srcj, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
* `gn` is the gradient defined on the last state,
"""
function treeverse_solve(s0, gn; param, srci, srcj, srcv, c, δ=20, logger=TreeverseLog())
    treeverse(x->treeverse_step!(x, param, srci, srcj, srcv, c),
        (x,g)->treeverse_grad(x, g[1], param, srci, srcj, srcv, g[2], c, g[3]),
        copy(s0), gn; δ=δ, N=param.NSTEP-1, f_inplace=true, logger=logger)
end
