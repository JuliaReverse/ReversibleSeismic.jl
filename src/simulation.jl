export AcousticPropagatorParams, AcousticPropagatorSolver

@with_kw mutable struct AcousticPropagatorParams
    # number of grids along x,y axis and time steps
    NX::Int64 = 101
    NY::Int64 = 641 
    NSTEP::Int64 = 2000 * 2 

    # size of grid cell and time step
    DELTAX::Float64 = 10.
    DELTAY::Float64 = 10.
    DELTAT::Float64 = 2.e-3 / 2

    # PML boundary conditon 
    USE_PML_XMIN::Bool = true
    USE_PML_XMAX::Bool = true
    USE_PML_YMIN::Bool = true
    USE_PML_YMAX::Bool = true
    NPOINTS_PML::Int64 = 12
    NPOWER::Int64 = 2
    damping_x::Union{Missing,Float64} = missing
    damping_y::Union{Missing,Float64} = missing
    Rcoef::Float64 = 0.001 # Relative reflection coefficient
    vp_ref::Float64 = 1000. 

    # Auxilliary Data
    Σx::Array{Float64} = []
    Σy::Array{Float64} = []
    IJ::Array{Int64} = []
    IJn::Array{Int64}  = []
    IJp::Array{Int64} = []
    IpJ::Array{Int64} = []
    IpJp::Array{Int64} = []
    IpJn::Array{Int64} = []
    InJ::Array{Int64} = []
    InJn::Array{Int64} = []
    InJp::Array{Int64} = []
    
    # display params
    IT_DISPLAY::Int64 = 0
end

function scatter_nd_ops(IJ, u, n)
    out = zeros(n)
    out[IJ] = u 
    out
end

function one_step(param::AcousticPropagatorParams, w, wold, φ, ψ, σ, τ, c)
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    IJ, IpJ, InJ, IJp, IJn, IpJp, IpJn, InJp, InJn =
        param.IJ, param.IpJ, param.InJ, param.IJp, param.IJn, param.IpJp, param.IpJn, param.InJp, param.InJn
        
    u = @. (2 - σ[IJ]*τ[IJ]*Δt^2 - 2*Δt^2/hx^2 * c[IJ] - 2*Δt^2/hy^2 * c[IJ]) * w[IJ] +
            c[IJ] * (Δt/hx)^2  *  (w[IpJ]+w[InJ]) +
            c[IJ] * (Δt/hy)^2  *  (w[IJp]+w[IJn]) +
            (Δt^2/(2hx))*(φ[IpJ]-φ[InJ]) +
            (Δt^2/(2hy))*(ψ[IJp]-ψ[IJn]) -
            (1 - (σ[IJ]+τ[IJ])*Δt/2) * wold[IJ] 
    u = @. u / (1 + (σ[IJ]+τ[IJ])/2*Δt)
    u = scatter_nd_ops(IJ, u, (param.NX+2)*(param.NY+2))
    φ = @. (1. -Δt*σ[IJ]) * φ[IJ] + Δt * c[IJ] * (τ[IJ] -σ[IJ])/2hx *  
        (u[IpJ]-u[InJ])
    ψ = @. (1. -Δt*τ[IJ]) * ψ[IJ] + Δt * c[IJ] * (σ[IJ] -τ[IJ])/2hy * 
        (u[IJp]-u[IJn])
    φ = scatter_nd_ops(IJ, φ, (param.NX+2)*(param.NY+2))
    ψ = scatter_nd_ops(IJ, ψ, (param.NX+2)*(param.NY+2))
    u, φ, ψ
end

function AcousticPropagatorSolver(param::AcousticPropagatorParams, srci::Int64, srcj::Int64, 
            srcv::Array{Float64, 1}, c::Array{Float64, 2})

    c = c[:]
    compute_PML_Params!(param)

    σij = param.Σx[:]
    τij = param.Σy[:]

    tu = zeros((param.NX+2)*(param.NY+2), param.NSTEP+1)
    tφ = zeros((param.NX+2)*(param.NY+2), param.NSTEP+1)
    tψ = zeros((param.NX+2)*(param.NY+2), param.NSTEP+1)

    for i = 3:param.NSTEP+1
        tu[:, i], tφ[:, i], tψ[:, i] = one_step(param, tu[:, i-1], tu[:, i-2], tφ[:, i-1], tψ[:, i-1], σij, τij, c)
        src_index = (srci - 1) * (param.NY+2) + srcj
        tu[src_index, i] += srcv[i-2]*param.DELTAT^2
    end

    tu
end

function getid2(a, b, nx, ny)
    idx = Int64[]
    for i = 1:length(b)
        for j = 1:length(a)
            push!(idx, (b[i]-1)*(nx+2) + a[j])
        end 
    end
    idx
end


function compute_PML_Params!(param::AcousticPropagatorParams)
    NX, NY = param.NX, param.NY
    # computing damping coefficient
    c, R = param.vp_ref, param.Rcoef
    Lx = param.NPOINTS_PML * param.DELTAX
    Ly = param.NPOINTS_PML * param.DELTAY
    param.damping_x = c/Lx*log(1/R)
    param.damping_y = c/Ly*log(1/R)
    # @show c, Lx, log(1/R), param.damping_x, param.damping_y


    X = (0:param.NX+1)*param.DELTAX
    Y = (0:param.NY+1)*param.DELTAY
    Σx = zeros(param.NX+2, param.NY+2)
    Σy = zeros(param.NX+2, param.NY+2)
    for i = 1:param.NX+2
        for j = 1:param.NY+2
            Σx[i,j], Σy[i,j] = pml_helper(X[i], Y[j], param)
        end
    end

    param.Σx = Σx
    param.Σy = Σy
    param.IJ = getid2(2:NX+1, 2:NY+1, NX, NY)
    param.IJn = getid2(2:NX+1, 1:NY, NX, NY)
    param.IJp = getid2(2:NX+1, 3:NY+2, NX, NY)
    param.IpJ = getid2(3:NX+2, 2:NY+1, NX, NY)
    param.IpJp = getid2(3:NX+2, 3:NY+2, NX, NY)
    param.IpJn = getid2(3:NX+2, 1:NY, NX, NY)
    param.InJ = getid2(1:NX, 2:NY+1, NX, NY)
    param.InJn = getid2(1:NX, 1:NY, NX, NY)
    param.InJp = getid2(1:NX, 3:NY+2, NX, NY)
    return param
end

"""
    pml_helper(x::Float64, y::Float64, param::AcousticPropagatorParams)

Computing the PML profile. 
"""
function pml_helper(x::Float64, y::Float64, param::AcousticPropagatorParams)
    outx = 0.0; outy = 0.0
    ξx = param.damping_x
    Lx = param.NPOINTS_PML * param.DELTAX
    if x<Lx && param.USE_PML_XMIN
        d = abs(Lx-x)
        outx = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    elseif x>param.DELTAX*(param.NX+1)-Lx && param.USE_PML_XMAX
        d = abs(x-(param.DELTAX*(param.NX+1)-Lx))
        outx = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    end

    ξy = param.damping_y
    Ly = param.NPOINTS_PML * param.DELTAY
    if y<Ly && param.USE_PML_YMIN
        d = abs(Ly-y)
        outy = ξy * (d/Ly - sin(2π*d/Ly)/(2π))
    elseif y>param.DELTAY*(param.NY+1)-Ly && param.USE_PML_YMAX
        d = abs(y-(param.DELTAY*(param.NY+1)-Ly))
        outy = ξy * (d/Ly - sin(2π*d/Ly)/(2π))
    end
    
    return outx, outy
end