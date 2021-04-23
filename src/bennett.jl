using NiLang.AD
export bennett_step!, bennett_solve
export SeismicState

struct SeismicState{MT}
    upre::MT
    u::MT
    φ::MT
    ψ::MT
    step::Base.RefValue{Int}
end
## solving gradient
function SeismicState(::Type{T}, nx::Int, ny::Int) where T
    SeismicState([zeros(T, nx+2, ny+2) for i=1:4]..., Ref(0))
end
Base.zero(x::SeismicState) = SeismicState(zero(x.upre), zero(x.u), zero(x.φ), zero(x.ψ), Ref(0))
Base.copy(x::SeismicState) = SeismicState(copy(x.upre), copy(x.u), copy(x.φ), copy(x.ψ), Ref(x.step[]))

@i function :(+=)(identity)(x::SeismicState, y::SeismicState)
    x.upre += y.upre
    x.u += y.u
    x.φ += y.φ
    x.ψ += y.ψ
    x.step[] += y.step[]
end

@i function bennett_step!(dest, src, param::AcousticPropagatorParams, srcloc, srcv, c)
    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    dest.upre += src.u
    dest.step[] += src.step[] + 1
    i_one_step!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c)
    dest.u[srcloc...] += srcv[dest.step[]] * d2
    ~@routine
end

function bennett_solve(s0, gn; param, src, srcv, c, k, N, logger=NiLang.BennettLog())
    # forward execution
    d = Dict(1=>s0)
    bennett!(bennett_step!, d, param, src, srcv, c; k=k, N=N, logger=logger)
    # backward execution
    y = d[N+1]
    y = SeismicState([GVar(getfield(y, field), getfield(gn, field)) for field in fieldnames(SeismicState)[1:end-1]]..., Ref(y.step[]))
    d = GVar(d)
    d[N+1] = y
    _, gs, _, _, gv, gc = (~bennett!)(bennett_step!, d, param, src, GVar(srcv), GVar(c); k=k, N=N, logger=logger)
    return grad(gs[1]), grad(gv), grad(gc)
end
