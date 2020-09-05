using KernelAbstractions

export i_solve_parallel!

function print_forward(str="")
    println(">>>>> $str >>>>>")
    str
end
function print_backward(str="")
    println("<<<<< $str <<<<<")
    str
end
@dual print_forward print_backward

# `DI/DJ ~ [0, 1]`, number of threads should be `(nx÷2) * (ny÷2)`.
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
        i ← inds[1]*2 + DI
        j ← inds[2]*2 + DJ
    end

    @routine @invcheckoff begin
        @inbounds begin
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
    end
    @inbounds u![i,j] += uij / σpτΔt_2
    ~@routine
    ~@routine
end

@i @kernel function i_one_step_kernel2!(Δt, hx, hy, u!, w, wold, φ!, φ0, ψ!, ψ0, σ, τ, c::AbstractMatrix{T}, vi::Val{DI}, vj::Val{DJ}) where {T,DI,DJ}
    @routine @invcheckoff begin
        inds ← @index(Global, NTuple)
        i ← inds[1]*2 + DI
        j ← inds[2]*2 + DJ
        @zeros Float64 Δt_hx Δt_hy
        Δt_hx += Δt / hx
        Δt_hy += Δt / hy
    end
    @routine @invcheckoff @inbounds begin
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
    @inbounds begin
        φ![i,j] -= σΔt * φ0[i,j]
        φ![i,j] -=  anc1 * dux
        ψ![i,j] -= τΔt * ψ0[i,j]
        ψ![i,j] += anc2 * duy
    end
    ~@routine
    ~@routine
end

@i function i_solve_parallel!(param::AcousticPropagatorParams, srci::Int, srcj::Int,
            srcv::AbstractArray{T, 1}, c::AbstractArray{T, 2},
            tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3};
            device, nthread::Int) where T
    for i = 3:param.NSTEP+1
        for (DI, DJ) in ((0, 0), (0, 1), (1, 0), (1, 1))
            @launchkernel device nthread (param.NX÷2, param.NY÷2) i_one_step_kernel1!(
                param.DELTAT, param.DELTAX, param.DELTAY, view(tu,:,:,i), view(tu,:,:,i-1), view(tu,:,:,i-2),
                view(tφ,:,:,i), view(tφ,:,:,i-1), view(tψ,:,:,i), view(tψ,:,:,i-1), param.Σx, param.Σy, c,
                Val(DI), Val(DJ))
        end
        for (DI, DJ) in ((0, 0), (0, 1), (1, 0), (1, 1))
            @launchkernel device nthread (param.NX÷2, param.NY÷2) i_one_step_kernel2!(
                param.DELTAT, param.DELTAX, param.DELTAY, view(tu,:,:,i), view(tu,:,:,i-1), view(tu,:,:,i-2),
                view(tφ,:,:,i), view(tφ,:,:,i-1), view(tψ,:,:,i), view(tψ,:,:,i-1), param.Σx, param.Σy, c,
                Val(DI), Val(DJ))
        end
        tu[srci, srcj, i] += srcv[i-2]*param.DELTAT^2
    end
end
