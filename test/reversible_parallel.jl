using Test
using ReversibleSeismic
using KernelAbstractions
using NiLang

"""
the reversible loss
"""
@i function i_loss_parallel!(out::T, param, srci, srcj, srcv::AbstractVector{T}, c::AbstractMatrix{T},
          tua::AbstractArray{T,3}, tφa::AbstractArray{T,3}, tψa::AbstractArray{T,3},
          tub::AbstractArray{T,3}, tφb::AbstractArray{T,3}, tψb::AbstractArray{T,3}) where T
     i_solve_parallel!(param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb; device=CPU(), nthread=2)
     for i=1:length(tub)
          out += tub[i] ^ 2
     end
end

function loss_parallel(c; na, nb)
     c = copy(c)
     nx, ny = size(c) .- 2
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2)  # nstep disabled

     tua = zeros(nx+2, ny+2, na)
     tφa = zeros(nx+2, ny+2, na)
     tψa = zeros(nx+2, ny+2, na)
     tub = zeros(nx+2, ny+2, 2nb)
     tφb = zeros(nx+2, ny+2, nb)
     tψb = zeros(nx+2, ny+2, nb)

     srci = nx ÷ 2
     srcj = ny ÷ 2
     srcv = Ricker(param, 100.0, 500.0)

     i_loss_parallel!(0.0, param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb)[1]
end

@testset "loss" begin
     nx = ny = 99
     na = 52
     nb = 41
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2)  # nstep disabled

     tua = zeros(nx+2, ny+2, na)
     tφa = zeros(nx+2, ny+2, na)
     tψa = zeros(nx+2, ny+2, na)
     tub = zeros(nx+2, ny+2, 2nb)
     tφb = zeros(nx+2, ny+2, nb)
     tψb = zeros(nx+2, ny+2, nb)

     c = 1000*ones(nx+2, ny+2)
     srci = nx ÷ 2
     srcj = ny ÷ 2
     srcv = Ricker(param, 100.0, 500.0)

     loss = i_loss_parallel!(0.0, param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb)[1]

     @test check_inv(i_loss_parallel!, (0.0, param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb); atol=1e-6)
     @test loss ≈ 10.931466822080788
end

"""
obtain gradients with NiLang.AD
"""
function getgrad_parallel(c::AbstractMatrix{T}; na::Int, nb::Int) where T
     c = copy(c)
     nx, ny = size(c) .- 2
     param = AcousticPropagatorParams(nx=nx, ny=ny,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2)

     tua = zeros(nx+2, ny+2, na)
     tφa = zeros(nx+2, ny+2, na)
     tψa = zeros(nx+2, ny+2, na)
     tub = zeros(nx+2, ny+2, 2nb)
     tφb = zeros(nx+2, ny+2, nb)
     tψb = zeros(nx+2, ny+2, nb)

     srci = size(c, 1) ÷ 2 - 1
     srcj = size(c, 2) ÷ 2 - 1
     srcv = Ricker(param, 100.0, 500.0)
     NiLang.AD.gradient(Val(1), i_loss_parallel!, (0.0, param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb))[6]
end

"""
obtain gradients numerically, for gradient checking.
"""
function getngrad_parallel(c::AbstractMatrix{T}, i, j; na::Int, nb::Int, δ=1e-4) where T
     c[i,j] += δ
     fpos = loss_parallel(c; na=na, nb=nb)
     c[i,j] -= 2δ
     fneg = loss_parallel(c; na=na, nb=nb)
     return (fpos - fneg)/2δ
end


@testset "gradient" begin
     nx = ny = 99
     c = 1000*ones(nx+2, ny+2)
     g4545 = getgrad_parallel(c; na=52, nb=41)[45,45]
     ng4545 = getngrad_parallel(c, 45, 45; na=52, nb=41, δ=1e-4)
     @test isapprox(g4545, ng4545; rtol=1e-2)
end
