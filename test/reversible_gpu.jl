using Test
using ReversibleSeismic
using KernelAbstractions
using NiLang
using NiLang.AD
using KernelAbstractions.CUDA
CUDA.allowscalar(false)

@testset "identity" begin
    x = CUDA.CuArray([1.0,2.0,3.0])
    y = CUDA.CuArray([5.0,2.0,3.0])
    @instr x += y
    @test Array(x) == [6.0, 4.0, 6.0]
    x = CUDA.CuArray([1.0,2.0,3.0])
    @instr @iforcescalar x[2] += 2.0
    @test Array(x) == [1.0, 4.0, 3.0]
end

"""
the reversible loss
"""
@i function i_loss_gpu!(out!::T, param, srci, srcj, srcv::Vector{T}, c::AbstractMatrix{T},
          tua::AbstractArray{T,3}, tφa::AbstractArray{T,3}, tψa::AbstractArray{T,3},
          tub::AbstractArray{T,3}, tφb::AbstractArray{T,3}, tψb::AbstractArray{T,3}) where T
    i_solve_parallel!(param, srci, srcj, srcv, c, tua, tφa, tψa, tub, tφb, tψb; device=CUDADevice(), nthreads=256)
    out! += sum((@skip! abs2), tub[:,:,end])
end

@i function i_loss_bennett_gpu!(out!::T, state, param, srci, srcj, srcv::Vector{T}, c::AbstractMatrix{T}; bennett_k=50, logger=NiLang.BennettLog()) where T
    bennett!((@const bennett_step!), state, bennett_k, 1, (@const param.NSTEP-1), param, srci, srcj, srcv, c; do_uncomputing=false, logger=logger)
    out! += sum((@skip! abs2), state[param.NSTEP].u)
end

function loss_gpu(c::AbstractMatrix{T}; na, nb) where T
    nx = size(c, 1) - 2
    ny = size(c, 2) - 2
    param = AcousticPropagatorParams(nx=nx, ny=ny,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2) |> togpu

    tua = CUDA.zeros(T, nx+2, ny+2, na)
    tφa = CUDA.zeros(T, nx+2, ny+2, na)
    tψa = CUDA.zeros(T, nx+2, ny+2, na)
    tub = CUDA.zeros(T, nx+2, ny+2, 2nb)
    tφb = CUDA.zeros(T, nx+2, ny+2, nb)
    tψb = CUDA.zeros(T, nx+2, ny+2, nb)

    srci = nx ÷ 2
    srcj = ny ÷ 2
    srcv = Ricker(param, 100.0, 500.0)

    loss = 0.0
    i_loss_gpu!(loss, param, srci, srcj, srcv, CuArray(c), tua, tφa, tψa, tub, tφb, tψb)[1]
end

function loss_bennett_gpu(c::AbstractMatrix{T}; nstep, usecuda=true) where T
    nx = size(c, 1) - 2
    ny = size(c, 2) - 2
    param = AcousticPropagatorParams(nx=nx, ny=ny,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)
    srci = nx ÷ 2
    srcj = ny ÷ 2
    srcv = Ricker(param, 100.0, 500.0)

    if usecuda
        state = Dict(1=>CuSeismicState(Float64, nx, ny))
        param = togpu(param)
        c = CuArray(c)
    else
        state = Dict(1=>SeismicState(Float64, nx, ny))
        c = copy(c)
    end
    i_loss_bennett_gpu!(0.0, state, param, srci, srcj, srcv, c)[1]
end

@testset "loss" begin
    T = Float64
    nx = ny = 99
    na = 52
    nb = 41
    param0 = AcousticPropagatorParams(nx=nx, ny=ny,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2)

    tua = CUDA.zeros(T, nx+2, ny+2, na)
    tφa = CUDA.zeros(T, nx+2, ny+2, na)
    tψa = CUDA.zeros(T, nx+2, ny+2, na)
    tub = CUDA.zeros(T, nx+2, ny+2, 2nb)
    tφb = CUDA.zeros(T, nx+2, ny+2, nb)
    tψb = CUDA.zeros(T, nx+2, ny+2, nb)

    c0 = 1000*CUDA.ones(T, nx+2, ny+2)
    srci = nx ÷ 2
    srcj = ny ÷ 2
    srcv0 = Ricker(param0, 100.0, 500.0)
    u = solve(param0, srci, srcj, copy(srcv0), Array(c0))[:,:,end]
    ls = sum(abs2, u)

    loss = i_loss_gpu!(0.0, togpu(param0), srci, srcj, copy(srcv0), copy(c0), copy(tua), copy(tφa), copy(tψa), copy(tub), copy(tφb), copy(tψb))[1]
    @test isapprox(loss, ls; rtol=1e-2)

    param0 = AcousticPropagatorParams(nx=nx, ny=ny,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2)
    s0 = CuSeismicState(Float64, nx, ny)
    state = Dict(1=>s0)
    loss_bg = i_loss_bennett_gpu!(0.0, copy(state), togpu(param0), srci, srcj, copy(srcv0), copy(c0); bennett_k=200)[1]

    s0 = CuSeismicState(Float64, nx, ny)
    state = Dict(1=>s0)
    loss_cpu = i_loss_bennett_gpu!(0.0, Dict(1=>SeismicState(Float64, nx, ny)), deepcopy(param0), srci, srcj, copy(srcv0), Array(c0))[1]
    @test loss_bg ≈ loss_cpu
    @test loss_bg ≈ ls
end

"""
obtain gradients with NiLang.AD
"""
function getgrad_gpu(c::AbstractMatrix{T}; na::Int, nb::Int) where T
    nx, ny = size(c) .- 2
    param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2) |> togpu

    c = c |> CuArray
    tua = CUDA.zeros(T, nx+2, ny+2, na)
    tφa = CUDA.zeros(T, nx+2, ny+2, na)
    tψa = CUDA.zeros(T, nx+2, ny+2, na)
    tub = CUDA.zeros(T, nx+2, ny+2, 2nb)
    tφb = CUDA.zeros(T, nx+2, ny+2, nb)
    tψb = CUDA.zeros(T, nx+2, ny+2, nb)

    srci = size(c, 1) ÷ 2 - 1
    srcj = size(c, 2) ÷ 2 - 1
    srcv = Ricker(param, 100.0, 500.0)
    NiLang.AD.gradient(Val(1), i_loss_gpu!, (0.0, param, srci, srcj, srcv,
               c, tua, tφa, tψa, tub, tφb, tψb))[6]
end

function getgrad_bennett_gpu(c::AbstractMatrix{T}; nstep) where T
    nx, ny = size(c) .- 2
    param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> togpu
    c = c |> CuArray
    srci = size(c, 1) ÷ 2 - 1
    srcj = size(c, 2) ÷ 2 - 1
    srcv = Ricker(param, 100.0, 500.0)
    s0 = CuSeismicState(Float64, nx, ny)
    state = Dict(1=>s0)
    NiLang.AD.gradient(Val(1), i_loss_bennett_gpu!, (0.0, state, param, srci, srcj, srcv, c))[7]
end

function getgrad_treeverse_gpu(c::AbstractMatrix{T}; nstep) where T
    nx, ny = size(c) .- 2
    param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> togpu
    c = c |> CuArray
    srci = size(c, 1) ÷ 2 - 1
    srcj = size(c, 2) ÷ 2 - 1
    srcv = Ricker(param, 100.0, 500.0)
    s0 = CuSeismicState(Float64, nx, ny)
    function gn(sn)
        g = zero(sn)
        g.u .+= 2 .* sn.u
        return (g, zero(srcv), zero(c))
    end
    treeverse_solve(s0, gn; param=param, srci=srci, srcj=srcj, srcv=srcv, c=c)[3]
end

"""
obtain gradients numerically, for gradient checking.
"""
function getngrad_gpu(c0::AbstractMatrix{T}, i, j; nstep::Int, δ=1e-4) where T
     c = c0 |> CuArray
     @forcescalar c[i,j] += δ
     fpos = loss_bennett_gpu(c; nstep)
     c = c0 |> CuArray
     @forcescalar c[i,j] -= δ
     fneg = loss_bennett_gpu(c; nstep)
     return (fpos - fneg)/2δ
end


@testset "gradient" begin
    nx = ny = 99
    na = 42
    nb = 51
    nstep=(na-2)*(nb-1)+2
    c = 1000*ones(Float64, nx+2, ny+2)
    g1 = Array(getgrad_gpu(c; na=na, nb=nb))
    g2 = Array(getgrad_bennett_gpu(c; nstep=nstep))
    g3 = Array(getgrad_treeverse_gpu(c; nstep=nstep))
    ng4545 = getngrad_gpu(c, 45, 45; nstep=nstep, δ=1e-3)
    @test isapprox(g1[45,45], ng4545; rtol=5e-2)
    @test isapprox(g2[45,45], ng4545; rtol=1e-2)
    @test isapprox(g3[45,45], ng4545; rtol=1e-2)
end
