@i function i_one_step(Δt, hx, hy, Nx, Ny, IJ, IpJ, InJ, IJp, IJn, IpJp, IpJn, InJp, InJn, w, wold, φ, ψ, σ, τ, c)
    u1 += 2
    u11 += σ[IJ]*τ[IJ]
    Δt2 += Δt ^ 2
    Δt_hx += Δt / hx
    Δt_hy += Δt / hy
    Δt_hx2 += Δt_hx ^ 2
    Δt_hy2 += Δt_hy ^ 2
    u1 -= u11 * Δt2
    twocIJ += 2 * c[IJ]
    u1 -= Δt_hx2 * 2cIJ
    u1 -= Δt_hy2 * 2cIJ
    u += u1 * w[IJ]
    u += c[IJ] * Δt_hx2  *  (w[IpJ]+w[InJ])
    u += c[IJ] * Δt_hy2  *  (w[IJp]+w[IJn])
    u += (Δt * Δt_hx/2)*(φ[IpJ]-φ[InJ])
    u += (Δt * Δt_hy/2)*(ψ[IJp]-ψ[IJn])
    u -= (1 - (σ[IJ]+τ[IJ])*Δt/2) * wold[IJ]

    u = u / (1 + (σ[IJ]+τ[IJ])/2*Δt)
    u = scatter_nd_ops(IJ, u, (NX+2)*(NY+2))
    φ += (1. -Δt*σ[IJ]) * φ[IJ]
    φ += Δt_hx * c[IJ] * (τ[IJ] -σ[IJ])/2 * (u[IpJ]-u[InJ])
    ψ += (1. -Δt*τ[IJ]) * ψ[IJ]
    ψ += Δt_hy * c[IJ] * (σ[IJ] -τ[IJ])/2 * (u[IJp]-u[IJn])
    NX += 2
    NY += 2
    φ = scatter_nd_ops(IJ, φ, NX * NY)
    ψ = scatter_nd_ops(IJ, ψ, NX * NY)
    #u, φ, ψ
end

