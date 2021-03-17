using .KernelAbstractions
using .KernelAbstractions.CUDA
using .CUDA
using NiLang.AD: GVar

export @iforcescalar, @forcescalar, togpu

function togpu(a::AcousticPropagatorParams{DIM}) where DIM
     AcousticPropagatorParams(a.NX, a.NY, a.NSTEP, a.DELTAX, a.DELTAY, a.DELTAT, CuArray(a.Σx), CuArray(a.Σy))
end

@inline function mp(out!, a)
    MinusEq(abs2)(out!, a)[2]
end

@inline function pp(out!, a)
   PlusEq(abs2)(out!, a)[2]
end
# define the gradient function
function (_::MinusEq{typeof(sum)})(out!::GVar{T}, ::typeof(abs2), x::AbstractArray{<:GVar{T}}) where T
    out! = GVar(out!.x-mapreduce(a->abs2(a.x), +, x; init=zero(T)), out!.g)
    x .= mp.(out!, x)
    out!, abs2, x
end

# define the gradient function
function (_::PlusEq{typeof(sum)})(out!::GVar{T}, ::typeof(abs2), x::AbstractArray{<:GVar{T}}) where {T}
    out! = chfield(out!, value, out!.x-sum(a->abs2(a.x), x))
    x .= pp.(out!, x)
    out!, abs2, x
end

function apply_take1(f, args...)
    f(args...)[1]
end

function apply_take2(f, args...)
    f(args...)[2]
end

function apply_take3(f, args...)
    f(args...)[3]
end

function NiLangCore.ibcast(f, a::CuArray, b::CuArray)
    f, apply_take1.(f, a, b), apply_take2.(f, a, b)
end

function NiLangCore.ibcast(f, a::CuArray, b::CuArray, c::CuArray)
    f, apply_take1.(f, a, b, c), apply_take2.(f, a, b, c), apply_take3.(f, a, b, c)
end

const CuSeismicState{MT} = SeismicState{MT} where MT<:CuArray

export CuSeismicState
function CuSeismicState(::Type{T}, nx::Int, ny::Int) where T
    SeismicState([CUDA.zeros(T, nx+2, ny+2) for i=1:4]..., 0)
end

function togpu(x::SeismicState)
    SeismicState([CuArray(t) for t in [x.upre, x.u, x.φ, x.ψ]]..., 0)
end

togpu(x::AbstractArray) = CuArray(x)

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

macro forcescalar(ex)
    quote
        x = $CUDA.GPUArrays.scalar_allowed[]
        $CUDA.allowscalar(true)
        res = $(esc(ex))
        $CUDA.GPUArrays.scalar_allowed[] = x
        res
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

@inline function cudiv(x::Int, y::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_y = min(max_threads ÷ threads_x, y)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (x, y) ./ threads)
    threads, blocks
end

function one_step!(param::AcousticPropagatorParams, u, w, wold, φ, ψ, σ, τ, c::CuArray)
    @inline function one_step_kernel1(u, w, wold, φ, ψ, σ, τ, c, Δt, Δtx, Δty)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x + 1
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y + 1
        Δtx2 = Δtx * Δtx
        Δty2 = Δty * Δty
        Dx = 0.5Δt*Δtx
        Dy = 0.5Δt*Δty
        @inbounds if i < size(c, 1) && j < size(c, 2)
            cij = c[i,j]
            δ = (σ[i,j]+τ[i,j])*Δt*0.5
            uij = (2 - σ[i,j]*τ[i,j]*(Δt*Δt) - 2*Δtx2 * cij - 2*Δty2 * cij) * w[i,j] +
                cij * Δtx2  *  (w[i+1,j]+w[i-1,j]) +
                cij * Δty2  *  (w[i,j+1]+w[i,j-1]) +
                Dx*(φ[i+1,j]-φ[i-1,j]) +
                Dy*(ψ[i,j+1]-ψ[i,j-1]) -
                (1 - δ) * wold[i,j] 
            u[i,j] = uij / (1 + δ)
        end
        return nothing
    end

    @inline function one_step_kernel2(u, φ, ψ, σ, τ, c, Δt, Δtx_2, Δty_2)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x + 1
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y + 1
        @inbounds if i < size(c, 1) && j < size(c, 2)
            φ[i,j] = (1-Δt*σ[i,j]) * φ[i,j] + Δtx_2 * c[i,j] * (τ[i,j] -σ[i,j]) *  
                (u[i+1,j]-u[i-1,j])
            ψ[i,j] = (1-Δt*τ[i,j]) * ψ[i,j] + Δty_2 * c[i,j] * (σ[i,j] -τ[i,j]) * 
                (u[i,j+1]-u[i,j-1])
        end
        return nothing
    end

    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
 
    threads, blocks = cudiv(param.NX, param.NY)
    @cuda threads=threads blocks=blocks one_step_kernel1(u, w, wold, φ, ψ, σ, τ, c, Δt, Δt/hx, Δt/hy)
    @cuda threads=threads blocks=blocks one_step_kernel2(u, φ, ψ, σ, τ, c, Δt, 0.5*Δt/hx, 0.5*Δt/hy)
    return nothing
end

