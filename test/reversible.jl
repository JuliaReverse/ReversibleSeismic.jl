using Test
using ReversibleSeismic
using NiLang

"""
the reversible loss
"""
@i function i_loss!(out::T, param, src, srcv::AbstractVector{T}, c::AbstractMatrix{T},
          tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
     i_solve!(param, src, srcv, c, tu, tφ, tψ)
     for i=1:size(tu, 1)
        for j=1:size(tu, 2)
            out += tu[i,j,end] ^ 2
        end
     end
end

"""
obtain gradients with NiLang.AD
"""
function getgrad(c::AbstractMatrix{T}; nstep::Int) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)

     c = copy(c)
     tu = zeros(T, size(c)..., nstep+1)
     tφ = zeros(T, size(c)..., nstep+1)
     tψ = zeros(T, size(c)..., nstep+1)

     src = size(c) .÷ 2 .- 1
     srcv = Ricker(param, 100.0, 500.0)
     NiLang.AD.gradient(Val(1), i_loss!, (0.0, param, src, srcv, c, tu, tφ, tψ))[end-3]
end

"""
obtain gradients numerically, for gradient checking.
"""
function getngrad(c::AbstractMatrix{T}, i, j; nstep::Int, δ=1e-4) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)

     c = copy(c)
     tu = zeros(T, size(c)..., nstep+1)
     tφ = zeros(T, size(c)..., nstep+1)
     tψ = zeros(T, size(c)..., nstep+1)

     src = size(c) .÷ 2 .- 1
     srcv = Ricker(param, 100.0, 500.0)
     c[i,j] += δ
     fpos = i_loss!(0.0, param, src, srcv, copy(c), copy(tu), copy(tφ), copy(tψ))[1]
     c[i,j] -= 2δ
     fneg = i_loss!(0.0, param, src, srcv, copy(c), copy(tu), copy(tφ), copy(tψ))[1]
     return (fpos - fneg)/2δ
end

@testset "loss" begin
     nx = ny = 100
     nstep = 2000
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)

     tu = zeros(nx+2, ny+2, nstep+1)
     tφ = zeros(nx+2, ny+2, nstep+1)
     tψ = zeros(nx+2, ny+2, nstep+1)

     c = 1000*ones(nx+2, ny+2)
     src = (nx ÷ 2, ny ÷ 2)
     srcv = Ricker(param, 100.0, 500.0)

     loss = i_loss!(0.0, param, src, srcv, c, tu, tφ, tψ)[1]
     @test check_inv(i_loss!, (0.0, param, src, srcv, c, tu, tφ, tψ); atol=1e-6)
     #@test loss ≈ 10.931466822080788
     @test loss ≈ 6.234873084873294e-5
end

@testset "gradient" begin
     nx = ny = 100
     nstep = 2000
     c = 1000*ones(nx+2, ny+2)
     g4545 = getgrad(c; nstep=nstep)[45,45]
     ng4545 = getngrad(c, 45, 45; nstep=nstep, δ=1e-4)
     @test isapprox(g4545, ng4545; rtol=1e-2)
end
