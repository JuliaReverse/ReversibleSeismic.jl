using CUDA

# `DI/DJ ~ [-1, 0, 1]`, number of threads should be `(nx÷3) * (ny÷3)`.
@i @inline function i_one_step_kernel1!(Δt, hx, hy, u!, w, wold, φ!, φ0, ψ!, ψ0, σ, τ, c::AbstractMatrix{T}, vi::Val{DI}, vj::Val{DJ}) where {T,DI,DJ}
    # update u!
    @routine @invcheckoff begin
        @zeros Float64 Δt2 Δt_hx Δt_hy Δt_hx2 Δt_hy2 Δt_2
        Δt2 += Δt ^ 2
        Δt_hx += Δt / hx
        Δt_hy += Δt / hy
        Δt_hx2 += Δt_hx ^ 2
        Δt_hy2 += Δt_hy ^ 2
        Δt_2 += Δt/2

        i ← ((blockIdx().x-1) * blockDim().x + threadIdx().x)*3 + DI - 1
        j ← ((blockIdx().y-1) * blockDim().y + threadIdx().y)*3 + DJ - 1
    end

    @invcheckoff @inbounds if i<size(u!, 1) && j<size(u!,2)
        @routine begin
            @zeros T σpτ σpτΔt_2 cΔt_hx2 cΔt_hy2 dwx dwy dφx dψy στ anc1 anc2 anc3 uij σij τij φ0ipj φ0imj ψ0ijp ψ0ijm wipj wimj wijp wijm woldij wij cij
            σij += σ[i,j]
            τij += τ[i,j]
            wipj += w[i+1,j]
            wimj += w[i-1,j]
            wijp += w[i,j+1]
            wijm += w[i,j-1]
            wij += w[i,j]
            woldij += wold[i,j]
            φ0ipj += φ0[i+1,j]
            φ0imj += φ0[i-1,j]
            ψ0ijp += ψ0[i,j+1]
            ψ0ijm += ψ0[i,j-1]
            cij += c[i,j]
            σpτ += σij + τij
            σpτΔt_2 += σpτ * Δt_2
            cΔt_hx2 += Δt_hx2 * cij
            cΔt_hy2 += Δt_hy2 * cij
            dwx += wipj + wimj
            dwy += wijp + wijm
            dφx += φ0ipj - φ0imj
            dψy += ψ0ijp - ψ0ijm
            στ += σij * τij

            anc1 += 2
            anc1 -= στ * Δt2
            anc1 -= 2 * cΔt_hx2
            anc1 -= 2 * cΔt_hy2
            anc2 += Δt_2 * Δt_hx
            anc3 += Δt_2 * Δt_hy
            uij += anc1 * wij
            uij += cΔt_hx2  *  dwx
            uij += cΔt_hy2  *  dwy
            uij += anc2 * dφx
            uij += anc3 * dψy
            uij -= woldij
            uij += σpτΔt_2 * woldij
            σpτΔt_2 += 1
        end
        u![i,j] += uij / σpτΔt_2
        ~@routine
    end
    ~@routine
end

@i @inline function i_one_step_kernel2!(Δt, hx, hy, u!, w, wold, φ!, φ0, ψ!, ψ0, σ, τ, c::AbstractMatrix{T}, vi::Val{DI}, vj::Val{DJ}) where {T,DI,DJ}
    @routine @invcheckoff begin
        i ← ((blockIdx().x-1) * blockDim().x + threadIdx().x)*3 + DI - 1
        j ← ((blockIdx().y-1) * blockDim().y + threadIdx().y)*3 + DJ - 1
        @zeros Float64 Δt_hx Δt_hy
        Δt_hx += Δt / hx
        Δt_hy += Δt / hy
    end
    @invcheckoff @inbounds if i<size(u!, 1) && j<size(u!,2)
        @routine begin
            @zeros T σmτ σmτ_2 dux duy cσmτ_2 σΔt τΔt anc1 anc2 σij τij uipj uimj uijp uijm φ0ij ψ0ij cij
            σij += σ[i,j]
            τij += τ[i,j]
            uipj += u![i+1,j]
            uimj += u![i-1,j]
            uijp += u![i,j+1]
            uijm += u![i,j-1]
            φ0ij += φ0[i,j]
            ψ0ij += ψ0[i,j]
            cij += c[i,j]
            σmτ += σij - τij
            σmτ_2 += σmτ / 2
            dux += uipj - uimj
            duy += uijp - uijm
            cσmτ_2 += cij * σmτ_2
            σΔt += Δt * σij
            τΔt += Δt * τij
            σΔt -= 1
            τΔt -= 1
            anc1 += Δt_hx * cσmτ_2
            anc2 += Δt_hy * cσmτ_2
        end
        φ![i,j] -= σΔt * φ0ij
        φ![i,j] -=  anc1 * dux
        ψ![i,j] -= τΔt * ψ0ij
        ψ![i,j] += anc2 * duy
        ~@routine
    end
    ~@routine
end

let
    exprs = [:((threads, blocks) ← cudiv(ceil(Int,param.NX/3), ceil(Int,param.NY/3)))]
    for KF in [:(i_one_step_kernel1!), :(i_one_step_kernel2!)]
        for (DI, DJ) in Base.Iterators.product((0,1,2), (0,1,2))
            push!(exprs, :(@cuda threads=threads blocks=blocks $KF(
                    param.DELTAT, param.DELTAX, param.DELTAY, u, w, wold,
                    φ, φ0, ψ, ψ0, param.Σx, param.Σy, c,
                    Val($DI), Val($DJ))))
        end
    end
    ex = Expr(:block, exprs...)
    @eval @i function i_one_step_parallel!(param::AcousticPropagatorParams, u, w, wold, φ, φ0, ψ, ψ0, c::AbstractMatrix{T}) where T
        $ex
    end
end

@i function bennett_step!(dest::T, src::T, param::AcousticPropagatorParams, srci, srcj, srcv, c) where T<:SeismicState{<:CuArray}
    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    dest.upre += src.u
    dest.step[] += src.step[] + 1
    @safe CUDA.synchronize()
    i_one_step_parallel!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c)
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
    #CUDA.memory_status()
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
