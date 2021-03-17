using .KernelAbstractions: CUDA
using .CUDA: CuArray, @cuda
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

