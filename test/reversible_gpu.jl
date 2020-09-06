using Test
using ReversibleSeismic
using KernelAbstractions
using NiLang
using NiLang.AD
using CUDA

"""
the reversible loss
"""
@i function i_loss_gpu!(out!::T, param, srci, srcj, srcv::Vector{T}, c::AbstractMatrix{T},
          tua::AbstractArray{T,3}, tφa::AbstractArray{T,3}, tψa::AbstractArray{T,3},
          tub::AbstractArray{T,3}, tφb::AbstractArray{T,3}, tψb::AbstractArray{T,3}) where T
     i_solve_parallel!(param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb; device=CUDADevice(), nthread=256)
     out! += sum((@skip! abs2), tub)
end

@testset "loss" begin
     T = Float64
     nx = ny = 99
     na = 52
     nb = 41
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2) |> cu

     tua = CUDA.zeros(T, nx+2, ny+2, na)
     tφa = CUDA.zeros(T, nx+2, ny+2, na)
     tψa = CUDA.zeros(T, nx+2, ny+2, na)
     tub = CUDA.zeros(T, nx+2, ny+2, 2nb)
     tφb = CUDA.zeros(T, nx+2, ny+2, nb)
     tψb = CUDA.zeros(T, nx+2, ny+2, nb)

     c = 1000*CUDA.ones(T, nx+2, ny+2)
     srci = nx ÷ 2
     srcj = ny ÷ 2
     srcv = Ricker(param, 100.0, 500.0)

     loss = 0.0
     @instr i_loss_gpu!(loss, param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb)
     @test loss ≈ 0.4317925530445345
     @instr ~i_loss_gpu!(loss, param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb)
     @test loss ≈ 0.0
     @test isapprox(Array(tub), zeros(nx+2, ny+2, 2nb), atol=1e-6)
end

"""
obtain gradients with NiLang.AD
"""
function getgrad_gpu(c::AbstractMatrix{T}; nstep::Int) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-1)*(nb-1)+2) |> cu

     c = c |> CuArray
     tua = CUDA.zeros(T, nx+2, ny+2, na)
     tφa = CUDA.zeros(T, nx+2, ny+2, na)
     tψa = CUDA.zeros(T, nx+2, ny+2, na)
     tub = CUDA.zeros(T, nx+2, ny+2, nb)
     tφb = CUDA.zeros(T, nx+2, ny+2, nb)
     tψb = CUDA.zeros(T, nx+2, ny+2, nb)

     srci = size(c, 1) ÷ 2 - 1
     srcj = size(c, 2) ÷ 2 - 1
     srcv = Ricker(param, 100.0, 500.0)
     NiLang.AD.gradient(Val(1), i_loss_gpu!, (0.0, param, srci, srcj, srcv,
               c, tua, tφa, tψa, tub, tφb, tψb))[6]
end

"""
obtain gradients numerically, for gradient checking.
"""
function getngrad_gpu(c::AbstractMatrix{T}, i, j; na::Int, nb::Int, δ=1e-4) where T
     c[i,j] += δ
     fpos = loss_gpu(c; na=na, nb=nb)
     c[i,j] -= 2δ
     fneg = loss_gpu(c; na=na, nb=nb)
     return (fpos - fneg)/2δ
end


@testset "gradient" begin
     nx = ny = 99
     nstep = 500
     c = 1000*ones(Float64, nx+2, ny+2)
     g4545 = getgrad_gpu(c; nstep=nstep)[45,45]
     ng4545 = getngrad_gpu(c, 45, 45; nstep=nstep, δ=1e-3)
     @test isapprox(g4545, ng4545; rtol=1e-2)
end
