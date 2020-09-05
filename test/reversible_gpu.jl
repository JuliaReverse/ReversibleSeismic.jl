using Test
using ReversibleSeismic
using KernelAbstractions
using NiLang
using CUDA

function CUDA.cu(a::AcousticPropagatorParams{DIM}) where DIM
     AcousticPropagatorParams(a.NX, a.NY, a.NSTEP, a.DELTAX, a.DELTAY, a.DELTAT, CuArray(a.Σx), CuArray(a.Σy))
end

"""
the reversible loss
"""
@i function i_loss_gpu!(out::T, param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T},
          tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
     i_solve_parallel!(param, srci, srcj, srcv, c, tu, tφ, tψ; device=CUDADevice(), nthread=256)
     for i=1:length(tu)
          out += tu[i] ^ 2
     end
end

@testset "loss" begin
     nx = ny = 100
     nstep = 2000
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> cu

     tu = zeros(nx+2, ny+2, nstep+1) |> CuArray
     tφ = zeros(nx+2, ny+2, nstep+1) |> CuArray
     tψ = zeros(nx+2, ny+2, nstep+1) |> CuArray

     c = 1000*ones(nx+2, ny+2) |> CuArray
     srci = nx ÷ 2
     srcj = ny ÷ 2
     srcv = Ricker(param, 100.0, 500.0) |> CuArray

     loss = i_loss_gpu!(0.0, param, srci, srcj, srcv, c, tu, tφ, tψ)[1]
     @test loss ≈ 10.931466822080788
     @test check_inv(i_loss_gpu!, (0.0, param, srci, srcj, srcv, c, tu, tφ, tψ); atol=1e-6)
end

"""
obtain gradients with NiLang.AD
"""
function getgrad_gpu(c::AbstractMatrix{T}; nstep::Int) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> cu

     c = copy(c)
     tu = zeros(T, size(c)..., nstep+1) |> CuArray
     tφ = zeros(T, size(c)..., nstep+1) |> CuArray
     tψ = zeros(T, size(c)..., nstep+1) |> CuArray

     srci = size(c, 1) ÷ 2 - 1
     srcj = size(c, 2) ÷ 2 - 1
     srcv = Ricker(param, 100.0, 500.0) |> CuArray
     NiLang.AD.gradient(Val(1), i_loss_gpu!, (0.0, param, srci, srcj, srcv, c, tu, tφ, tψ))[end-3]
end

"""
obtain gradients numerically, for gradient checking.
"""
function getngrad_gpu(c::AbstractMatrix{T}, i, j; nstep::Int, δ=1e-4) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> cu

     c = copy(c)
     tu = zeros(T, size(c)..., nstep+1) |> CuArray
     tφ = zeros(T, size(c)..., nstep+1) |> CuArray
     tψ = zeros(T, size(c)..., nstep+1) |> CuArray

     srci = size(c, 1) ÷ 2 - 1
     srcj = size(c, 2) ÷ 2 - 1
     srcv = Ricker(param, 100.0, 500.0) |> CuArray
     c[i,j] += δ
     fpos = i_loss_gpu!(0.0, param, srci, srcj, srcv, copy(c), copy(tu), copy(tφ), copy(tψ))[1]
     c[i,j] -= 2δ
     fneg = i_loss_gpu!(0.0, param, srci, srcj, srcv, copy(c), copy(tu), copy(tφ), copy(tψ))[1]
     return (fpos - fneg)/2δ
end


@testset "gradient" begin
     nx = ny = 100
     nstep = 200
     c = 1000*ones(nx+2, ny+2)
     g4545 = getgrad_gpu(c; nstep=nstep)[45,45]
     ng4545 = getngrad_gpu(c, 45, 45; nstep=nstep, δ=1e-4)
     @test isapprox(g4545, ng4545; rtol=1e-2)
end
