export solve_detector, bennett_step_detector!, treeverse_grad_detector,
    treeverse_solve_detector, i_loss_bennett_detector!
function solve_detector(param::AcousticPropagatorParams, srci::Int, srcj::Int, 
            srcv::Array{Float64, 1}, c::Array{Float64, 2}, detector_locs::AbstractVector)
    slices = zeros(length(detector_locs), param.NSTEP-1)
    tupre = zeros(param.NX+2, param.NY+2)
    tu = zeros(param.NX+2, param.NY+2)
    tφ = zeros(param.NX+2, param.NY+2)
    tψ = zeros(param.NX+2, param.NY+2)

    for i = 3:param.NSTEP+1
        tu_ = zeros(param.NX+2, param.NY+2)
        one_step!(param, tu_, tu, tupre, tφ, tψ, param.Σx, param.Σy, c)
        tu, tupre = tu_, tu
        tu[srci, srcj] += srcv[i-2]*param.DELTAT^2
        slices[:,i-2] .= tu[detector_locs]
    end
    slices
end

@i function bennett_step_detector!(_dest::T, _src::T, param::AcousticPropagatorParams, srci, srcj, srcv, c, target_pulses, detector_locs) where T<:Glued
    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
        temp ← zeros(eltype(c), length(detector_locs))
        (data_dest,) ← @unsafe_destruct _dest
        (data_src,) ← @unsafe_destruct _src
        (dloss, dest) ← @unsafe_destruct data_dest
        (sloss, src) ← @unsafe_destruct data_src
    end
    dest.upre += src.u
    dest.step += src.step + 1
    i_one_step!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c)
    dest.u[srci, srcj] += srcv[dest.step] * d2
    @routine begin
        temp += target_pulses[:,dest.step]
        temp -= dest.u[detector_locs]
    end
    dloss += sloss
    for i=1:size(target_pulses, 1)
        dloss += temp[i]^2
    end
    ~@routine
    ~@routine
end

"""
    treeverse_solve(s0; param, srci, srcj, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
"""
function treeverse_solve_detector(s0; param, srci, srcj, srcv, c, target_pulses, detector_locs, δ=20, logger=TreeverseLog())
    f = x->treeverse_step_detector(x, param, srci, srcj, srcv, c, target_pulses, detector_locs)
    function gf(x, g)
        if g === nothing
            g = (Glued(one(x.data[1]),zero(x.data[2])), zero(srcv), zero(c))
        end
        treeverse_grad_detector(x, g[1], param, srci, srcj, srcv, g[2], c, g[3], target_pulses, detector_locs)
    end
    treeverse(f, gf,
        copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=false, logger=logger)
end

function treeverse_grad_detector(x_, g_, param, srci, srcj, srcv, gsrcv, c, gc, target_pulses, detector_locs)
    lg, g = g_.data
    ly, y = treeverse_step_detector(x_, param, srci, srcj, srcv, c, target_pulses, detector_locs).data
    #g.u[detector_locs] .+= 2 .* (y.u[detector_locs] .-  target_pulses[y.step])
    gt = SeismicState([GVar(getfield(y, field), getfield(g, field)) for field in fieldnames(SeismicState)[1:end-1]]..., y.step)
    _, gs, _, _, _, gv, gc2 = (~bennett_step_detector!)(Glued(GVar(ly, lg), gt), GVar(x_), param, srci, srcj, GVar(srcv, gsrcv), GVar(c, gc), GVar(target_pulses), detector_locs)
    (grad(gs), grad(gv), grad(gc2))
end

@i function i_loss_bennett_detector!(out, state, param, srci, srcj, srcv, c, target_pulses, detector_locs; k, logger=NiLang.BennettLog())
    bennett!((@const bennett_step_detector!), state, k, 1, (@const param.NSTEP-1), param, srci, srcj, srcv, c, target_pulses, detector_locs; do_uncomputing=false, logger=logger)
    out += state[param.NSTEP].data.:1
end

function treeverse_step_detector(s_, param, srci, srcj, srcv, c, target_pulses, detector_locs)
    l, s = s_.data
    unext, φ, ψ = zero(s.u), copy(s.φ), copy(s.ψ)
    ReversibleSeismic.one_step!(param, unext, s.u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(s.u, unext, φ, ψ, s.step+1)
    s2.u[srci, srcj] += srcv[s2.step]*param.DELTAT^2
    l += sum(abs2, target_pulses[:,s2.step] - s2.u[detector_locs])
    return Glued(l, s2)
end


