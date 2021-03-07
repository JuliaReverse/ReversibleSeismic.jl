export i_solve!

# do not wrap fields
NiLang.AD.GVar(param::AcousticPropagatorParams) = param
NiLang.AD.grad(param::AcousticPropagatorParams) = nothing

@i function i_one_step!(param::AcousticPropagatorParams, u, w, wold, φ, φ0, ψ, ψ0, c::AbstractMatrix{T}) where T
    @routine @invcheckoff begin
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
                σpτ += param.Σx[i,j] + param.Σy[i,j]
                σpτΔt_2 += σpτ * Δt_2
                cΔt_hx2 += Δt_hx2 * c[i,j]
                cΔt_hy2 += Δt_hy2 * c[i,j]
                dwx += w[i+1,j] + w[i-1,j]
                dwy += w[i,j+1] + w[i,j-1]
                dφx += φ0[i+1,j] - φ0[i-1,j]
                dψy += ψ0[i,j+1] - ψ0[i,j-1]
                στ += param.Σx[i,j] * param.Σy[i,j]

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
                σmτ += param.Σx[i,j] - param.Σy[i,j]
                σmτ_2 += σmτ / 2
                dux += u[i+1,j] - u[i-1,j]
                duy += u[i,j+1] - u[i,j-1]
                cσmτ_2 += c[i,j] * σmτ_2
                σΔt += Δt * param.Σx[i,j]
                τΔt += Δt * param.Σy[i,j]
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

@i function i_solve!(param::AcousticPropagatorParams, srci::Int, srcj::Int,
            srcv::Array{T, 1}, c::Array{T, 2},
            tu::Array{T,3}, tφ::Array{T,3}, tψ::Array{T,3}) where T

    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    for i = 3:param.NSTEP+1
        i_one_step!(param, tu |> subarray(:,:,i), tu |> subarray(:,:,i-1), tu |> subarray(:,:,i-2),
            tφ |> subarray(:,:,i), tφ |> subarray(:,:,i-1), tψ |> subarray(:,:,i), tψ |> subarray(:,:,i-1), c)
        tu[srci, srcj, i] += srcv[i-2]* d2
    end
    ~@routine
end
