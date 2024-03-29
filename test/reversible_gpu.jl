using Test
using ReversibleSeismic
using NiLang
using NiLang.AD
using CUDA

CUDA.allowscalar(false)

@testset "identity" begin
    x = CUDA.CuArray([1.0,2.0,3.0])
    y = CUDA.CuArray([5.0,2.0,3.0])
    @instr x += y
    @test Array(x) == [6.0, 4.0, 6.0]
end

@i function i_loss_bennett_gpu!(out!::T, state, param, src, srcv::Vector{T}, c::AbstractMatrix{T}; bennett_k=50, logger=NiLang.BennettLog()) where T
    bennett!((@const bennett_step!), state, bennett_k, 1, (@const param.NSTEP-1), param, src, srcv, c; do_uncomputing=false, logger=logger)
    out! += sum((@skip! abs2), state[param.NSTEP].u)
end

function loss_bennett_gpu(c::AbstractMatrix{T}; nstep, usecuda=true) where T
    nx = size(c, 1) - 2
    ny = size(c, 2) - 2
    param = AcousticPropagatorParams(nx=nx, ny=ny,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)
    src = (nx ÷ 2, ny ÷ 2)
    srcv = Ricker(param, 100.0, 500.0)

    if usecuda
        state = Dict(1=>CuSeismicState(Float64, nx, ny))
        param = togpu(param)
        c = CuArray(c)
    else
        state = Dict(1=>SeismicState(Float64, nx, ny))
        c = copy(c)
    end
    i_loss_bennett_gpu!(0.0, state, param, src, srcv, c)[1]
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
    src = (nx ÷ 2, ny ÷ 2)
    srcv0 = Ricker(param0, 100.0, 500.0)
    u = solve(param0, src, copy(srcv0), Array(c0))[:,:,end]
    ls = sum(abs2, u)

    param0 = AcousticPropagatorParams(nx=nx, ny=ny,
         Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=(na-2)*(nb-1)+2)
    s0 = CuSeismicState(Float64, nx, ny)
    state = Dict(1=>s0)
    loss_bg = i_loss_bennett_gpu!(0.0, copy(state), togpu(param0), src, copy(srcv0), copy(c0); bennett_k=200)[1]

    s0 = CuSeismicState(Float64, nx, ny)
    state = Dict(1=>s0)
    loss_cpu = i_loss_bennett_gpu!(0.0, Dict(1=>SeismicState(Float64, nx, ny)), deepcopy(param0), src, copy(srcv0), Array(c0))[1]
    @test loss_bg ≈ loss_cpu
    @test loss_bg ≈ ls
end

function getgrad_bennett_gpu(c::AbstractMatrix{T}; nstep) where T
    nx, ny = size(c) .- 2
    param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> togpu
    c = c |> CuArray
    src = size(c) .÷ 2 .- 1
    srcv = Ricker(param, 100.0, 500.0)
    s0 = CuSeismicState(Float64, nx, ny)
    state = Dict(1=>s0)
    NiLang.AD.gradient(Val(1), i_loss_bennett_gpu!, (0.0, state, param, src, srcv, c))[6]
end

function getgrad_treeverse_gpu(c::AbstractMatrix{T}; nstep) where T
    nx, ny = size(c) .- 2
    param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep) |> togpu
    c = c |> CuArray
    src = size(c) .÷ 2 .- 1
    srcv = Ricker(param, 100.0, 500.0)
    s0 = CuSeismicState(Float64, nx, ny)
    function gn(sn)
        g = zero(sn)
        g.u .+= 2 .* sn.u
        return (g, zero(srcv), zero(c))
    end
    treeverse_solve(s0, gn; param=param, src=src, srcv=srcv, c=c)[2][3]
end

"""
obtain gradients numerically, for gradient checking.
"""
function getngrad_gpu(c0::AbstractMatrix{T}, i, j; nstep::Int, δ=1e-4) where T
     c = c0 |> CuArray
     c[ReversibleSeismic.SafeIndex(i,j)] += δ
     fpos = loss_bennett_gpu(c; nstep)
     c = c0 |> CuArray
     c[ReversibleSeismic.SafeIndex(i,j)] -= δ
     fneg = loss_bennett_gpu(c; nstep)
     return (fpos - fneg)/2δ
end


@testset "gradient" begin
    nx = ny = 99
    na = 42
    nb = 51
    nstep=(na-2)*(nb-1)+2
    c = 1000*ones(Float64, nx+2, ny+2)
    g2 = Array(getgrad_bennett_gpu(c; nstep=nstep))
    g3 = Array(getgrad_treeverse_gpu(c; nstep=nstep))
    ng4545 = getngrad_gpu(c, 45, 45; nstep=nstep, δ=1e-3)
    @test isapprox(g2[45,45], ng4545; rtol=1e-2)
    @test isapprox(g3[45,45], ng4545; rtol=1e-2)
end
