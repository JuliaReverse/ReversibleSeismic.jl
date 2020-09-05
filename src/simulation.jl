export AcousticPropagatorParams, AcousticPropagatorSolver

struct AcousticPropagatorParams{DIM}
    # number of grids along x,y axis and time steps
    NX::Int64
    NY::Int64 
    NSTEP::Int64

    # size of grid cell and time step
    DELTAX::Float64
    DELTAY::Float64
    DELTAT::Float64

    # PML boundary conditon 
    NPOINTS_PML::Int64
    NPOWER::Int64
    damping_x::Float64
    damping_y::Float64
    Rcoef::Float64 # Relative reflection coefficient
    vp_ref::Float64

    # Auxilliary Data
    Σx::Array{Float64,DIM}
    Σy::Array{Float64,DIM}
end

function AcousticPropagatorParams(; nx, ny, nstep, dx, dy, dt,
        npoints_PML=12, npower=2,
        Rcoef=0.001, vp_ref=1000.0,
        USE_PML_XMAX = true,
        USE_PML_XMIN = true,
        USE_PML_YMAX = true,
        USE_PML_YMIN = true)
    # computing damping coefficient
    Lx = npoints_PML * dx
    Ly = npoints_PML * dy
    damping_x = vp_ref/Lx*log(1/Rcoef)
    damping_y = vp_ref/Ly*log(1/Rcoef)


    param = AcousticPropagatorParams{2}(nx, ny, nstep, dx, dy, dt,
        npoints_PML, npower, damping_x, damping_y,
        Rcoef, vp_ref, zeros(nx+2, ny+2), zeros(nx+2, ny+2)
    )

    X = (0.0:param.NX+1) * dx
    Y = (0.0:param.NY+1) * dy
    for i = 1:param.NX+2
        for j = 1:param.NY+2
            param.Σx[i,j], param.Σy[i,j] = pml_helper(X[i], Y[j], param;
                USE_PML_XMAX = USE_PML_XMAX,
                USE_PML_XMIN = USE_PML_XMIN,
                USE_PML_YMAX = USE_PML_YMAX,
                USE_PML_YMIN = USE_PML_YMIN)
        end
    end
    return param
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

    tu = zeros(param.NX+2, param.NY+2, param.NSTEP+1)
    tφ = zeros(param.NX+2, param.NY+2)
    tψ = zeros(param.NX+2, param.NY+2)

    for i = 3:param.NSTEP+1
        tu[:,:,i] .= one_step!(param, tu[:,:,i-1], tu[:,:,i-2], tφ, tψ, param.Σx, param.Σy, c)
        tu[srci, srcj, i] += srcv[i-2]*param.DELTAT^2
    end

    tu
end

"""
    pml_helper(x::Float64, y::Float64, param::AcousticPropagatorParams)

Computing the PML profile. 
"""
function pml_helper(x::Float64, y::Float64, param::AcousticPropagatorParams;
        USE_PML_XMAX = true,
        USE_PML_XMIN = true,
        USE_PML_YMAX = true,
        USE_PML_YMIN = true,
    )
    outx = 0.0; outy = 0.0
    ξx = param.damping_x
    Lx = param.NPOINTS_PML * param.DELTAX
    if x<Lx && USE_PML_XMIN 
        d = abs(Lx-x)
        outx = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    elseif x>param.DELTAX*(param.NX+1)-Lx && USE_PML_XMAX
        d = abs(x-(param.DELTAX*(param.NX+1)-Lx))
        outx = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    end

    ξy = param.damping_y
    Ly = param.NPOINTS_PML * param.DELTAY
    if y<Ly && USE_PML_YMIN
        d = abs(Ly-y)
        outy = ξy * (d/Ly - sin(2π*d/Ly)/(2π))
    elseif y>param.DELTAY*(param.NY+1)-Ly && USE_PML_YMAX
        d = abs(y-(param.DELTAY*(param.NY+1)-Ly))
        outy = ξy * (d/Ly - sin(2π*d/Ly)/(2π))
    end
    
    return outx, outy
end