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
        tu[src_index, i] += srcv[i]*param.DELTAT^2
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

function compute_PML_Params(param::ElasticPropagatorParams)
    ## define profile of absorption in PML region
    NPOINTS_PML = param.NPOINTS_PML
    NPOWER = param.NPOWER
    NX = param.NX
    NY = param.NY
    DELTAX = param.DELTAX
    DELTAY = param.DELTAY
    DELTAT = param.DELTAT
    K_MAX_PML = param.K_MAX_PML
    ALPHA_MAX_PML = param.ALPHA_MAX_PML
    USE_PML_XMIN = param.USE_PML_XMIN
    USE_PML_XMAX = param.USE_PML_XMAX
    USE_PML_YMIN = param.USE_PML_YMIN
    USE_PML_YMAX = param.USE_PML_YMAX
    Rcoef = param.Rcoef
    
    thickness_PML_x = NPOINTS_PML * DELTAX
    thickness_PML_y = NPOINTS_PML * DELTAY
    
    # reflection coefficient
    d0_x = - (NPOWER + 1) * param.vp_ref * log(Rcoef) / (2. * thickness_PML_x)
    d0_y = - (NPOWER + 1) * param.vp_ref * log(Rcoef) / (2. * thickness_PML_y)
    
    xoriginleft = thickness_PML_x
    xoriginright = (NX-0.5)*DELTAX - thickness_PML_x
    
    a_x = zeros(NX)
    a_y = zeros(NY)
    a_x_half = zeros(NX)
    a_y_half = zeros(NY)
    b_x = zeros(NX)
    b_y = zeros(NY)
    b_x_half = zeros(NX)
    b_y_half = zeros(NY)
    d_x = zeros(NX)
    d_y = zeros(NY)
    d_x_half = zeros(NX)
    d_y_half = zeros(NY)
    K_x = ones(NX)
    K_y = ones(NY)
    K_x_half = ones(NX)
    K_y_half = ones(NY)
    alpha_x = zeros(NX)
    alpha_y = zeros(NY)
    alpha_x_half = zeros(NX)
    alpha_y_half = zeros(NY)
    
    for i = 1:NX
        xval = DELTAX * (i-1)
        if (USE_PML_XMIN)
            
            abscissa_in_PML = xoriginleft - xval
            if (abscissa_in_PML >= 0.0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized^NPOWER
                K_x[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x[i] = ALPHA_MAX_PML * abscissa_normalized
            end
            
            abscissa_in_PML = xoriginleft - (xval + DELTAX/2.)
            if (abscissa_in_PML >= 0.0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized^NPOWER
                K_x_half[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x_half[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x_half[i] = ALPHA_MAX_PML * abscissa_normalized
            end
            
        end
        
        if (USE_PML_XMAX)
            abscissa_in_PML = xval - xoriginright
            if (abscissa_in_PML >= 0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized^NPOWER
                K_x[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x[i] = ALPHA_MAX_PML * abscissa_normalized
            end
            
            abscissa_in_PML = xval + DELTAX/2.0 - xoriginright
            if (abscissa_in_PML >= 0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized^NPOWER
                K_x_half[i] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_x_half[i] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_x_half[i] = ALPHA_MAX_PML * abscissa_normalized
            end
        end
        
        if (alpha_x[i] < 0) 
            alpha_x[i] = 0
        end
        if (alpha_x_half[i] < 0) 
            alpha_x_half[i] = 0
        end
        
        b_x[i] = exp(- (d_x[i] / K_x[i] + alpha_x[i]) * DELTAT)
        b_x_half[i] = exp(- (d_x_half[i] / K_x_half[i] + alpha_x_half[i]) * DELTAT)
        
        if (abs(d_x[i]) > 1e-6) 
            a_x[i] = d_x[i] * (b_x[i] - 1.) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i]))
        end
        if (abs(d_x_half[i]) > 1e-6) 
            a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i]))
        end
    end
    
    yoriginbottom = thickness_PML_y
    yorigintop = (NY-0.5)*DELTAY - thickness_PML_y
    
    for j = 1:NY
        
        yval = DELTAY * (j-1)
        if param.USE_PML_YMIN
            abscissa_in_PML = yoriginbottom - yval
            if (abscissa_in_PML >= 0.0)
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y[j] = d0_y * abscissa_normalized^NPOWER
                K_y[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y[j] = ALPHA_MAX_PML * abscissa_normalized
            end
            abscissa_in_PML = yoriginbottom - (yval + DELTAY/2.)
            if abscissa_in_PML >= 0
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y_half[j] = d0_y * abscissa_normalized^NPOWER
                K_y_half[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y_half[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y_half[j] = ALPHA_MAX_PML * abscissa_normalized
            end
        end
        
        if param.USE_PML_YMAX
            abscissa_in_PML = yval - yorigintop
            if abscissa_in_PML >= 0
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y[j] = d0_y * abscissa_normalized^NPOWER
                K_y[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y[j] = ALPHA_MAX_PML * abscissa_normalized
            end
            
            abscissa_in_PML = yval + DELTAY/2. - yorigintop
            if abscissa_in_PML >= 0
                abscissa_normalized = abscissa_in_PML / thickness_PML_y
                d_y_half[j] = d0_y * abscissa_normalized^NPOWER
                K_y_half[j] = 1. + (K_MAX_PML - 1.) * abscissa_normalized^NPOWER
                # alpha_y_half[j] = ALPHA_MAX_PML * (1. - abscissa_normalized)
                alpha_y_half[j] = ALPHA_MAX_PML * abscissa_normalized
            end
        end
        
        b_y[j] = exp(- (d_y[j] / K_y[j] + alpha_y[j]) * DELTAT)
        b_y_half[j] = exp(- (d_y_half[j] / K_y_half[j] + alpha_y_half[j]) * DELTAT)
        if abs(d_y[j]) > 1e-6
            a_y[j] = d_y[j] * (b_y[j] - 1.) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
        end
        if (abs(d_y_half[j]) > 1e-6) 
            a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))
        end
    end
    # @show norm(d_x-d_y)

    ax = [a_x'; a_x_half']
    bx = [b_x'; b_x_half']
    kx = [K_x'; K_x_half']
    ay = [a_y'; a_y_half']
    by = [b_y'; b_y_half']
    ky = [K_y'; K_y_half']
    
    
    return ax,bx,kx,ay,by,ky
    
end
