using .KernelAbstractions
using .CUDAKernels
using .CUDAKernels.CUDA

# `DI/DJ ~ [-1, 0, 1]`, number of threads should be `(nx÷3) * (ny÷3)`.
@i @kernel function i_one_step_kernel1!(Δt, hx, hy, u!, w, wold, φ!, φ0, ψ!, ψ0, σ, τ, c::AbstractMatrix{T}, vi::Val{DI}, vj::Val{DJ}) where {T,DI,DJ}
    # update u!
    @routine @invcheckoff begin
        @zeros Float64 Δt2 Δt_hx Δt_hy Δt_hx2 Δt_hy2 Δt_2
        Δt2 += Δt ^ 2
        Δt_hx += Δt / hx
        Δt_hy += Δt / hy
        Δt_hx2 += Δt_hx ^ 2
        Δt_hy2 += Δt_hy ^ 2
        Δt_2 += Δt/2

        inds ← @index(Global, NTuple)
        i ← inds[1]*3 + DI - 1
        j ← inds[2]*3 + DJ - 1
    end

    @invcheckoff @inbounds if i<size(u!, 1) && j<size(u!,2)
        @routine begin
            @zeros T σpτ σpτΔt_2 cΔt_hx2 cΔt_hy2 dwx dwy dφx dψy στ anc1 anc2 anc3 uij
            σpτ += σ[i,j] + τ[i,j]
            σpτΔt_2 += σpτ * Δt_2
            cΔt_hx2 += Δt_hx2 * c[i,j]
            cΔt_hy2 += Δt_hy2 * c[i,j]
            dwx += w[i+1,j] + w[i-1,j]
            dwy += w[i,j+1] + w[i,j-1]
            dφx += φ0[i+1,j] - φ0[i-1,j]
            dψy += ψ0[i,j+1] - ψ0[i,j-1]
            στ += σ[i,j] * τ[i,j]

            anc1 += 2
            anc1 -= στ * Δt2
            anc1 -= 2 * cΔt_hx2
            anc1 -= 2 * cΔt_hy2
            anc2 += Δt_2 * Δt_hx
            anc3 += Δt_2 * Δt_hy
            uij += anc1 * w[i,j]
            uij += cΔt_hx2  *  dwx
            uij += cΔt_hy2  *  dwy
            uij += anc2 * dφx
            uij += anc3 * dψy
            uij -= wold[i,j]
            uij += σpτΔt_2 * wold[i,j]
            σpτΔt_2 += 1
        end
        u![i,j] += uij / σpτΔt_2
        ~@routine
    end
    ~@routine
end

@i @kernel function i_one_step_kernel2!(Δt, hx, hy, u!, w, wold, φ!, φ0, ψ!, ψ0, σ, τ, c::AbstractMatrix{T}, vi::Val{DI}, vj::Val{DJ}) where {T,DI,DJ}
    @routine @invcheckoff begin
        inds ← @index(Global, NTuple)
        i ← inds[1]*3 + DI - 1
        j ← inds[2]*3 + DJ - 1
        @zeros Float64 Δt_hx Δt_hy
        Δt_hx += Δt / hx
        Δt_hy += Δt / hy
    end
    @invcheckoff @inbounds if i<size(u!, 1) && j<size(u!,2)
        @routine begin
            @zeros T σmτ σmτ_2 dux duy cσmτ_2 σΔt τΔt anc1 anc2
            σmτ += σ[i,j] - τ[i,j]
            σmτ_2 += σmτ / 2
            dux += u![i+1,j] - u![i-1,j]
            duy += u![i,j+1] - u![i,j-1]
            cσmτ_2 += c[i,j] * σmτ_2
            σΔt += Δt * σ[i,j]
            τΔt += Δt * τ[i,j]
            σΔt -= 1
            τΔt -= 1
            anc1 += Δt_hx * cσmτ_2
            anc2 += Δt_hy * cσmτ_2
        end
        φ![i,j] -= σΔt * φ0[i,j]
        φ![i,j] -=  anc1 * dux
        ψ![i,j] -= τΔt * ψ0[i,j]
        ψ![i,j] += anc2 * duy
        ~@routine
    end
    ~@routine
end

let
    exprs = []
    for KF in [:(i_one_step_kernel1!), :(i_one_step_kernel2!)]
        for (DI, DJ) in Base.Iterators.product((0,1,2), (0,1,2))
            push!(exprs, :(@launchkernel device nthreads (ceil(Int,param.NX/3), ceil(Int,param.NY/3)) $KF(
                    param.DELTAT, param.DELTAX, param.DELTAY, u, w, wold,
                    φ, φ0, ψ, ψ0, param.Σx, param.Σy, c,
                    Val($DI), Val($DJ))))
        end
    end
    ex = Expr(:block, exprs...)
    @eval @i function i_one_step_parallel!(param::AcousticPropagatorParams, u, w, wold, φ, φ0, ψ, ψ0, c::AbstractMatrix{T}; device, nthreads) where T
        $ex
    end
end

@i function bennett_step!(dest::T, src::T, param::AcousticPropagatorParams, srci, srcj, srcv, c; nthreads=256) where T<:SeismicState{<:CuArray}
    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    dest.upre += src.u
    dest.step[] += src.step[] + 1
    @safe CUDA.synchronize()
    i_one_step_parallel!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c; device=CUDADevice(), nthreads=nthreads)
    dest.u[SafeIndex(srci, srcj)] += srcv[dest.step[]] * d2
    ~@routine
end

function treeverse_step(s::CuSeismicState, param, srci, srcj, srcv, c::CuMatrix)
    unext, u, φ, ψ = zero(s.u), copy(s.u), copy(s.φ), copy(s.ψ)
    one_step!(param, unext, u, s.upre, φ, ψ, param.Σx, param.Σy, c)
    s2 = SeismicState(u, unext, φ, ψ, Ref(s.step[]+1))
    s2.u[SafeIndex(srci, srcj)] += srcv[s2.step[]]*param.DELTAT^2
    return s2
end

function treeverse_grad(x::CuSeismicState, g::CuSeismicState, param, srci, srcj, srcv, gsrcv, c::CuMatrix, gc::CuMatrix)
    println("gradient: $(x.step[]+1) -> $(x.step[])")
    CUDA.memory_status()
    y = treeverse_step(x, param, srci, srcj, srcv, c)  # this function is not inplace!
    gt = SeismicState([GVar(getfield(y, field), getfield(g, field)) for field in fieldnames(SeismicState)[1:end-1]]..., Ref(y.step[]))
    x = GVar(x)
    srcv = GVar(srcv, gsrcv)
    c = GVar(c, gc)
    (~bennett_step!)(gt, x, param, srci, srcj, srcv, c)
    gv = grad(srcv)
    gc2 = grad(c)
    gs = grad(x)
    (gs, gv, gc2)
end
