using NiLang

export i_solve!

@i function i_one_step!(param::AcousticPropagatorParams, u, w, wold, φ, φ0, ψ, ψ0, σ, τ, c::AbstractMatrix{T}) where T
    @routine begin
        @zeros Float64 Δt hx hy Δt2 Δt_hx Δt_hy Δt_hx2 Δt_hy2 Δt_2
        Δt += param.DELTAT
        hx += param.DELTAX
        hy += param.DELTAY
        Δt2 += Δt ^ 2
        Δt_hx += Δt / hx
        Δt_hy += Δt / hy
        Δt_hx2 += Δt_hx ^ 2
        Δt_hy2 += Δt_hy ^ 2
        Δt_2 += Δt/2
    end
 
    @invcheckoff @inbounds for j=2:param.NY+1
        for i=2:param.NX+1
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
            u[i,j] += uij / σpτΔt_2
            ~@routine
        end
    end
    @invcheckoff @inbounds for j=2:param.NY+1
        for i=2:param.NX+1
            @routine begin
                @zeros T σmτ σmτ_2 dux duy cσmτ_2 σΔt τΔt anc1 anc2
                σmτ += σ[i,j] - τ[i,j]
                σmτ_2 += σmτ / 2
                dux += u[i+1,j] - u[i-1,j]
                duy += u[i,j+1] - u[i,j-1]
                cσmτ_2 += c[i,j] * σmτ_2
                σΔt += Δt * σ[i,j]
                τΔt += Δt * τ[i,j]
                σΔt -= 1
                τΔt -= 1
                anc1 += Δt_hx * cσmτ_2
                anc2 += Δt_hy * cσmτ_2
            end
            φ[i,j] -= σΔt * φ0[i,j]
            φ[i,j] -=  anc1 * dux
            ψ[i,j] -= τΔt * ψ0[i,j]
            ψ[i,j] += anc2 * duy
            ~@routine
        end
    end
    ~@routine
end

@i function i_solve!(param::AcousticPropagatorParams, srci::Int64, srcj::Int64, 
            srcv::Array{Float64, 1}, c::Array{T, 2},
            tu::Array{T,3}, tφ::Array{T,3}, tψ::Array{T,3}) where T

    for i = 3:param.NSTEP+1
        i_one_step!(param, view(tu,:,:,i), view(tu,:,:,i-1), view(tu,:,:,i-2),
            view(tφ,:,:,i), view(tφ,:,:,i-1), view(tψ,:,:,i), view(tψ,:,:,i-1), param.Σx, param.Σy, c)
        tu[srci, srcj, i] += srcv[i-2]*param.DELTAT^2
    end
end