export solve_detector, bennett_step_detector!, treeverse_grad_detector,
    treeverse_solve_detector, i_loss_bennett_detector!, solve_detector2

export Glued, RK4, ODESolve, ODEStep,
    i_ODESolve, i_ODEStep, ODELog, checkpointed_neuralode

struct Glued{T}
    data::T
end
Glued(args...) = Glued(args)

Base.zero(c::Glued) = Glued(zero.(c.data))
Base.copy(c::Glued) = Glued(copy.(c.data))
@generated function Base.zero(::Type{Glued{T}}) where T
    :(Glued($([zero(t) for t in T.types]...)))
end

@inline function Base.:(+)(a::Glued{T}, b::Glued{T}) where T
    Glued{T}(a.data .+ b.data)
end

@inline function Base.:(/)(a::Glued{T}, b::Real) where T
    Glued{T}(a.data ./ b)
end

@inline function Base.:(*)(a::Real, b::Glued{T}) where T
    Glued{T}(a .* b.data)
end

function zero_similar(arr::AbstractArray{T}, size...) where T
    zeros(T, size...)
end

function solve_detector(param::AcousticPropagatorParams, src, 
            srcv::AbstractArray{Float64, 1}, c::AbstractArray{Float64, 2}, detector_locs::AbstractVector)
    slices = zero_similar(c, length(detector_locs), param.NSTEP-1)
    tupre = zero_similar(c, param.NX+2, param.NY+2)
    tu = zero_similar(c, param.NX+2, param.NY+2)
    tφ = zero_similar(c, param.NX+2, param.NY+2)
    tψ = zero_similar(c, param.NX+2, param.NY+2)

    for i = 1:param.NSTEP-1
        tu_ = zero_similar(c, param.NX+2, param.NY+2)
        one_step!(param, tu_, tu, tupre, tφ, tψ, param.Σx, param.Σy, c)
        tu, tupre = tu_, tu
        tu[SafeIndex(src)] += srcv[i]*param.DELTAT^2
        slices[:,i] .= tu[detector_locs]
    end
    slices
end

@i function bennett_step_detector!(_dest::T, _src::T, param::AcousticPropagatorParams, srcloc, srcv, c, target_pulses, detector_locs) where T<:Glued
    @routine @invcheckoff begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
        temp ← zeros(eltype(c), length(detector_locs))
        (data_dest,) ← @unsafe_destruct _dest
        (data_src,) ← @unsafe_destruct _src
        (dloss, dest) ← @unsafe_destruct data_dest
        (sloss, src) ← @unsafe_destruct data_src
    end
    dest.upre += src.u
    dest.step[] += src.step[] + 1
    i_one_step!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c)
    dest.u[srcloc...] += srcv[dest.step[]] * d2
    @routine begin
        temp += target_pulses[:,dest.step[]]
        temp -= dest.u[detector_locs]
    end
    dloss += sloss
    for i=1:size(target_pulses, 1)
        dloss += temp[i]^2
    end
    ~@routine
    ~@routine
end

struct GradientCache{TS,TP,TV,TP2}
    x::TS
    y::TS
    c::TP
    srcv::TV
    target_pulses::TP2
end

"""
    treeverse_solve(s0; param, src, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
"""
function treeverse_solve_detector(s0; param, src, srcv, c, target_pulses, detector_locs, δ=20, logger=TreeverseLog())
    f = x->treeverse_step_detector(x, param, src, srcv, c, target_pulses, detector_locs)
    res = []
    gcache = GradientCache(GVar(s0.data[2]), GVar(s0.data[2]), GVar(c), GVar(srcv), GVar(target_pulses))
    function gf(x, g)
        if g === nothing
            y = f(x)
            push!(res, y)
            g = (Glued(one(x.data[1]),zero(x.data[2])), zero(srcv), zero(c))
        end
        gy, gsrcv, gc = g
        treeverse_grad_detector(x, gy, param, src, srcv, gsrcv, c, gc, target_pulses, detector_locs, gcache)
    end
    g = treeverse(f, gf,
        copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=false, logger=logger)
    res[], g
end

function treeverse_grad_detector(x_, g_, param, src, srcv, gsrcv, c, gc, target_pulses, detector_locs, gcache)
    lx, x = x_.data
    lg, g = g_.data
    println("gradient: $(x.step[]+1) -> $(x.step[])")
    #CUDA.memory_status()
    ly, y = treeverse_step_detector(x_, param, src, srcv, c, target_pulses, detector_locs).data

    # fit data into cache
    gcache.c .= GVar.(c, gc)
    gcache.srcv .= GVar.(srcv, gsrcv)
    for field in fieldnames(SeismicState)[1:end-1]
        getfield(gcache.y, field) .= GVar.(getfield(y, field), getfield(g, field))
        getfield(gcache.x, field) .= GVar.(getfield(x, field))
    end
    gcache.x.step[] = x.step[]
    gcache.y.step[] = y.step[]

    # compute
    _, gs, _, _, gv, gc2 = (~bennett_step_detector!)(Glued(GVar(ly, lg), gcache.y), Glued(GVar(lx), gcache.x), param, src, gcache.srcv, gcache.c, gcache.target_pulses, detector_locs)

    # get gradients from the cache
    gc .= grad(gc2)
    gsrcv .= grad.(gv)
    for field in fieldnames(SeismicState)[1:end-1]
        getfield(g, field) .= grad.(getfield(gs.data[2], field))
    end

    return (Glued(grad(gs.data[1]), g), gsrcv, gc)
end

@i function i_loss_bennett_detector!(out, state, param, src, srcv, c, target_pulses, detector_locs; k, logger=NiLang.BennettLog())
    bennett!((@const bennett_step_detector!), state, k, 1, (@const param.NSTEP-1), param, src, srcv, c, target_pulses, detector_locs; do_uncomputing=false, logger=logger)
    out += state[param.NSTEP].data.:1
end

function treeverse_step_detector(s_, param, src, srcv, c, target_pulses, detector_locs)
    l, s = s_.data
    unext, φ, ψ = zero(s.u), copy(s.φ), copy(s.ψ)
    ReversibleSeismic.one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(copy(s.u), unext, φ, ψ, Ref(s.step[]+1))
    s2.u[SafeIndex(src)] += srcv[s2.step[]]*param.DELTAT^2
    l += sum(abs2.(target_pulses[:,s2.step[]] .- s2.u[detector_locs]))
    return Glued(l, s2)
end

function solve_detector2(param::AcousticPropagatorParams, srcloc, 
            srcv::Array{T, 1}, c::AbstractArray{T, 2}, target_pulses, detector_locs) where T
    src = Glued(0.0, (c isa CuArray ? CuSeismicState : SeismicState)(T, param.NX, param.NY))
    for i=1:param.NSTEP-1
        dest = _zero(src)
        src, = bennett_step_detector!(dest, src, param, srcloc, srcv, c, target_pulses, detector_locs)
    end
    return src
end
