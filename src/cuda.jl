using .CUDA
using NiLang.AD: GVar

function CUDA.cu(a::AcousticPropagatorParams{DIM}) where DIM
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
