using .KernelAbstractions

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
        i ← inds[1]*3 + DI - 1
        j ← inds[2]*3 + DJ - 1
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
            tua::AbstractArray{T,3}, tφa::AbstractArray{T,3}, tψa::AbstractArray{T,3},
            tub::AbstractArray{T,3}, tφb::AbstractArray{T,3}, tψb::AbstractArray{T,3};
            device, nthread::Int) where T
    @safe @assert param.NX%3 == 0 && param.NY%3 == 0 "NX and NY must be multiple of 3, got $(param.NX) and $(param.NY)"
    @safe @assert size(tψa)[1] == param.NX+2 && size(tψa)[2] == param.NY+2
    @safe @assert size(tψa) == size(tφa) == size(tua)
    @safe @assert size(tψb) == size(tφb) == (size(tub, 1), size(tub, 2), size(tub,3)÷2)
    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    for b = 1:size(tub, 3)÷2-1
        @routine begin
            # load data from the stack top of B to A
            tua[:,:,1] .+= tub[:,:,2b-1]
            tua[:,:,2] .+= tub[:,:,2b]
            tφa[:,:,2] .+= tφb[:,:,b]
            tψa[:,:,2] .+= tψb[:,:,b]
            @safe CUDA.synchronize()  #! need to sync!
            for a = 3:size(tua, 3)
                for (DI, DJ) in Base.Iterators.product((0,1,2), (0,1,2))
                    @launchkernel device nthread (param.NX÷3, param.NY÷3) i_one_step_kernel1!(
                        param.DELTAT, param.DELTAX, param.DELTAY, view(tua,:,:,a), view(tua,:,:,a-1), view(tua,:,:,a-2),
                        view(tφa,:,:,a), view(tφa,:,:,a-1), view(tψa,:,:,a), view(tψa,:,:,a-1), param.Σx, param.Σy, c,
                        Val(DI), Val(DJ))
                end
                for (DI, DJ) in Base.Iterators.product((0,1,2), (0,1,2))
                    @launchkernel device nthread (param.NX÷3, param.NY÷3) i_one_step_kernel2!(
                        param.DELTAT, param.DELTAX, param.DELTAY, view(tua,:,:,a), view(tua,:,:,a-1), view(tua,:,:,a-2),
                        view(tφa,:,:,a), view(tφa,:,:,a-1), view(tψa,:,:,a), view(tψa,:,:,a-1), param.Σx, param.Σy, c,
                        Val(DI), Val(DJ))
                end
                tua[srci, srcj, a] += srcv[(b-1)*(size(tua,3)-2) + a] * d2
            end
        end
        # copy the stack top of A to B
        tub[:,:,2b+1] .+= tua[:,:,end-1]
        tub[:,:,2b+2] .+= tua[:,:,end]
        tφb[:,:,b+1] .+= tφa[:,:,end]
        tψb[:,:,b+1] .+= tψa[:,:,end]
        ~@routine
        @safe tua .= 0.0  # avoid the accumulation of rounding errors!
        @safe tφa .= 0.0
        @safe tψa .= 0.0
        @safe GC.gc()
    end
    ~@routine
end
