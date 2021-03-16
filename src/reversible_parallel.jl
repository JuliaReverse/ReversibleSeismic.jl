using .KernelAbstractions
using .KernelAbstractions.CUDA

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

NiLangCore.assign_ex(x, y; invcheck) = @show (x, y)
export @iforcescalar
macro iforcescalar(ex)
    x = gensym()
    esc(quote
        $x ← $(CUDA.GPUArrays).ScalarAllowed
        $(NiLang.SWAP)($(CUDA.GPUArrays).scalar_allowed[], $x)
        $(ex)
        $(NiLang.SWAP)($(CUDA.GPUArrays).scalar_allowed[], $x)
        $x → $(CUDA.GPUArrays).ScalarAllowed
    end)
end

export @forcescalar
macro forcescalar(ex)
    quote
        x = $CUDA.GPUArrays.scalar_allowed[]
        $CUDA.allowscalar(true)
        $(esc(ex))
        $CUDA.GPUArrays.scalar_allowed[] = x
    end
end

@i function addkernel(target, source)
    @invcheckoff b ← (blockIdx().x-1) * blockDim().x + threadIdx().x
    @invcheckoff if (b <= length(target), ~)
        @inbounds target[b] += source[b]
    end
    @invcheckoff b → (blockIdx().x-1) * blockDim().x + threadIdx().x
end

@i function :(+=)(identity)(target::CuArray, source::CuArray)
    @safe @assert length(target) == length(source)
    @cuda threads=256 blocks=ceil(Int,length(target)/256) addkernel(target, source)
end

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

@i function i_one_step_parallel!(param::AcousticPropagatorParams, u, w, wold, φ, φ0, ψ, ψ0, c::AbstractMatrix{T}; device, nthreads) where T
    for (DI, DJ) in Base.Iterators.product((0,1,2), (0,1,2))
        @launchkernel device nthreads (param.NX÷3, param.NY÷3) i_one_step_kernel1!(
            param.DELTAT, param.DELTAX, param.DELTAY, u, w, wold,
            φ, φ0, ψ, ψ0, param.Σx, param.Σy, c,
            Val(DI), Val(DJ))
    end
    for (DI, DJ) in Base.Iterators.product((0,1,2), (0,1,2))
        @launchkernel device nthreads (param.NX÷3, param.NY÷3) i_one_step_kernel2!(
            param.DELTAT, param.DELTAX, param.DELTAY, u, w, wold,
            φ, φ0, ψ, ψ0, param.Σx, param.Σy, c,
            Val(DI), Val(DJ))
    end
end

@i function i_solve_parallel!(param::AcousticPropagatorParams, srci::Int, srcj::Int,
            srcv::AbstractArray{T, 1}, c::AbstractArray{T, 2},
            tua::AbstractArray{T,3}, tφa::AbstractArray{T,3}, tψa::AbstractArray{T,3},
            tub::AbstractArray{T,3}, tφb::AbstractArray{T,3}, tψb::AbstractArray{T,3};
            device, nthreads::Int) where T
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
                i_one_step_parallel!(param, tua|>subarray(:,:,a), tua|>subarray(:,:,a-1), tua|>subarray(:,:,a-2),
                        tφa|>subarray(:,:,a), tφa|>subarray(:,:,a-1), tψa|>subarray(:,:,a), tψa|>subarray(:,:,a-1), c;
                        device=device, nthreads=nthreads)
                @iforcescalar tua[srci, srcj, a] += srcv[(b-1)*(size(tua,3)-2) + a] * d2
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

@i function bennett_step!(dest::T, src::T, param::AcousticPropagatorParams, srci, srcj, srcv, c; nthreads=256) where T<:SeismicState{<:CuArray}
    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    dest.upre += src.u
    dest.step += src.step + 1
    @safe CUDA.synchronize()
    i_one_step_parallel!(param, dest.u, src.u, src.upre,
        dest.φ, src.φ, dest.ψ, src.ψ, c; device=CUDADevice(), nthreads=nthreads)
    @iforcescalar dest.u[srci, srcj] += srcv[dest.step] * d2
    ~@routine
end

export CuSeismicState
function CuSeismicState(::Type{T}, nx::Int, ny::Int) where T
    SeismicState([CUDA.zeros(T, nx+2, ny+2) for i=1:4]..., 0)
end