export bennett_step!
using NiLang: i_addto!

struct SeismicState{MT}
    upre::MT
    u::MT
    φ::MT
    ψ::MT
    i::Int
end

Base.zero(x::SeismicState) = SeismicState(zero(x.upre), zero(x.u), zero(x.φ), zero(x.ψ), zero(x.i))
Base.copy(x::SeismicState) = SeismicState(copy(x.upre), copy(x.u), copy(x.φ), copy(x.ψ), x.i)

@i function NiLang.i_addto!(x::SeismicState, y::SeismicState)
    i_addto!(x.upre, y.upre)
    i_addto!(x.u, y.u)
    i_addto!(x.φ, y.φ)
    i_addto!(x.ψ, y.ψ)
    x.i += y.i
end

@i function bennett_step!(dest, src, param::AcousticPropagatorParams, srci, srcj, srcv, c)
    i_addto!(dest.upre, src.u)
    dest.i += src.i + 1
    ReversibleSeismic.i_one_step!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c)
    dest.u[srci, srcj] += srcv[dest.i]*param.DELTAT^2
end