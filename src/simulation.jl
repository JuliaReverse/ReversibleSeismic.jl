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
    
    # display params
    IT_DISPLAY::Int64 = 0
end

function one_step!(param::AcousticPropagatorParams, w, wold, φ, ψ, σ, τ, c)
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
    u = zeros(param.NX+2, param.NY+2)
    c = reshape(c, param.NX+2, param.NY+2)
 
    for j=2:param.NY+1, i=2:param.NX+1
        uij = (2 - σ[i,j]*τ[i,j]*Δt^2 - 2*Δt^2/hx^2 * c[i,j] - 2*Δt^2/hy^2 * c[i,j]) * w[i,j] +
            c[i,j] * (Δt/hx)^2  *  (w[i+1,j]+w[i-1,j]) +
            c[i,j] * (Δt/hy)^2  *  (w[i,j+1]+w[i,j-1]) +
            (Δt^2/(2hx))*(φ[i+1,j]-φ[i-1,j]) +
            (Δt^2/(2hy))*(ψ[i,j+1]-ψ[i,j-1]) -
            (1 - (σ[i,j]+τ[i,j])*Δt/2) * wold[i,j] 
        u[i,j] = uij / (1 + (σ[i,j]+τ[i,j])/2*Δt)
    end
    for j=2:param.NY+1, i=2:param.NX+1
        φ[i,j] = (1. -Δt*σ[i,j]) * φ[i,j] + Δt * c[i,j] * (τ[i,j] -σ[i,j])/2hx *  
            (u[i+1,j]-u[i-1,j])
        ψ[i,j] = (1-Δt*τ[i,j]) * ψ[i,j] + Δt * c[i,j] * (σ[i,j] -τ[i,j])/2hy * 
            (u[i,j+1]-u[i,j-1])
    end
    u
end

function AcousticPropagatorSolver(param::AcousticPropagatorParams, srci::Int64, srcj::Int64, 
            srcv::Array{Float64, 1}, c::Array{Float64, 2})

    c = reshape(c, param.NX+2, param.NY+2)
    compute_PML_Params!(param)

    σij = reshape(param.Σx[:], param.NX+2, param.NY+2)
    τij = reshape(param.Σy[:], param.NX+2, param.NY+2)

    tu = zeros(param.NX+2, param.NY+2, param.NSTEP+1)
    tφ = zeros(param.NX+2, param.NY+2)
    tψ = zeros(param.NX+2, param.NY+2)

    for i = 3:param.NSTEP+1
        tu[:,:,i] .= one_step!(param, tu[:,:,i-1], tu[:,:,i-2], tφ, tψ, σij, τij, c)
        tu[srci, srcj, i] += srcv[i-2]*param.DELTAT^2
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