### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 5720d27a-f176-11ea-35ec-efcc84b4e2a9
using NiLang, Plots

# ╔═╡ d87a3c74-f17a-11ea-1ce7-31659781efee
using ReversibleSeismic: i_one_step!, AcousticPropagatorParams, Ricker

# ╔═╡ b0d4d10c-f191-11ea-2f84-1b19dcb85d6a
using Optim

# ╔═╡ 701eab34-f17f-11ea-17e9-fbfa0b20ab76
md"# To define a loss"

# ╔═╡ 937e8aa6-f17a-11ea-1d5d-6f495a29d584
@i function i_solve!(param::AcousticPropagatorParams, src,
            srcv::Array{T, 1}, c::Array{T, 2},
            tu::Array{T,3}, tφ::Array{T,3}, tψ::Array{T,3}) where T

    @routine begin
        d2 ← zero(param.DELTAT)
        d2 += param.DELTAT^2
    end
    for i = 3:param.NSTEP+1
        i_one_step!(param, view(tu,:,:,i), view(tu,:,:,i-1), view(tu,:,:,i-2),
            view(tφ,:,:,i), view(tφ,:,:,i-1), view(tψ,:,:,i), view(tψ,:,:,i-1), param.Σx, param.Σy, c)
        tu[src[1], src[2], i] += srcv[i-2] * d2
    end
    ~@routine
end

# ╔═╡ 453bb1f2-f18e-11ea-086f-d18ed1055e0f
nstep = 1500

# ╔═╡ 0fa21e04-f192-11ea-03db-7d8b07acfa91
c = 700 * (1 .+ sin.(LinRange(0, 5π, 82))' .* cos.(LinRange(0, 3π, 82)));

# ╔═╡ b4a4159a-f178-11ea-07fc-059d9c6119cb
"""
the reversible loss
"""
@i function i_loss!(out::T, param, src, srcv::AbstractVector{T}, c0::AbstractMatrix{T}, tu::AbstractArray{T,3}, tφ::AbstractArray{T,3}, tψ::AbstractArray{T,3}) where T
	@routine begin
		c ← zero(c0)
		for i=1:length(c0)
			c[i] += abs(c0[i])
		end
	end
    i_solve!(param, src, srcv, c, tu, tφ, tψ)
	out -= tu[size(tu,1)÷2,size(tu,2)÷2+20,end]
	~@routine
end

# ╔═╡ 66d4e274-f180-11ea-0831-a946570eca63
function loss(c; nstep)
	nx, ny = size(c) .- 2
	param = AcousticPropagatorParams(nx=nx, ny=ny,
	Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)

 	tu = zeros(nx+2, ny+2, nstep+1)
 	tφ = zeros(nx+2, ny+2, nstep+1)
 	tψ = zeros(nx+2, ny+2, nstep+1)
	
 	src = (nx ÷ 2, ny ÷ 2)
 	srcv = Ricker(param, 100.0, 500.0)

 	res = i_loss!(0.0, param, src, srcv, copy(c), tu, tφ, tψ)
	res[1], res[7]
end

# ╔═╡ a8f5a20a-f195-11ea-167f-7d408ebe0366
heatmap(c)

# ╔═╡ 3b10da96-f18d-11ea-16fa-bf8437f54903
tu_seq = loss(c, nstep=nstep)[2];

# ╔═╡ cd4c73b6-f17c-11ea-08be-5bf044147749
ani = @animate for i=1:100:nstep
	heatmap(tu_seq[2:end-1,2:end-1,i], clim=(-0.005, 0.005))
end

# ╔═╡ cea59342-f17d-11ea-11c3-5ddc3062b120
gif(ani, fps=5)

# ╔═╡ 03c0a192-f198-11ea-3a75-f7b26638fc6a
heatmap(tu_seq[:,:,end])

# ╔═╡ 5b9381f2-f17f-11ea-3fb0-c5f5314c3c69
md"# Obtaining gradients"

# ╔═╡ 0681da22-f17a-11ea-1fca-a7d4004af233
"""
obtain gradients with NiLang.AD
"""
function getgrad(c::AbstractMatrix{T}; nstep::Int) where T
     param = AcousticPropagatorParams(nx=size(c,1)-2, ny=size(c,2)-2,
          Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)

     c = copy(c)
     tu = zeros(T, size(c)..., nstep+1)
     tφ = zeros(T, size(c)..., nstep+1)
     tψ = zeros(T, size(c)..., nstep+1)

     src = size(c) .÷ 2 .- 1
     srcv = Ricker(param, 100.0, 500.0)
     NiLang.AD.gradient(Val(1), i_loss!, (0.0, param, src, srcv, c, tu, tφ, tψ))[end-3]
end;

# ╔═╡ 5f8782ca-f17a-11ea-082a-a98ea089b700
gradc = getgrad(c; nstep=nstep);

# ╔═╡ 7bea3116-f18d-11ea-19d8-e5ef06dd114a
heatmap(gradc)

# ╔═╡ 16504abe-f180-11ea-33b6-bfb2937f0846
res = optimize(c->(@show loss(c, nstep=nstep)[1]), (g, c)->(g.=getgrad(c; nstep=nstep)), c, BFGS(), Optim.Options(iterations=5, g_tol=1e-30, f_tol=1e-30))

# ╔═╡ 07a889aa-f193-11ea-3a0d-1bacdd6cc2cb
heatmap(abs.(res.minimizer) - abs.(c))

# ╔═╡ 5a869532-f18d-11ea-2f3e-2b5886e7fa70
tu_seq2 = loss(res.minimizer, nstep=nstep)[2];

# ╔═╡ 5cd02706-f181-11ea-217e-c167fc0a4164
ani2 = @animate for i=1:nstep
	heatmap(tu_seq2[2:end-1,2:end-1,i], clim=(-0.005, 0.005))
end every 100

# ╔═╡ af3428da-f181-11ea-0ea8-f9bfabc320d3
gif(ani2, fps=5)

# ╔═╡ f36eea06-f192-11ea-003e-b91e794f3a6f
heatmap(tu_seq2[2:end-1,2:end-1,end], clim=(-0.005, 0.005))

# ╔═╡ Cell order:
# ╠═5720d27a-f176-11ea-35ec-efcc84b4e2a9
# ╠═d87a3c74-f17a-11ea-1ce7-31659781efee
# ╟─701eab34-f17f-11ea-17e9-fbfa0b20ab76
# ╠═937e8aa6-f17a-11ea-1d5d-6f495a29d584
# ╠═b4a4159a-f178-11ea-07fc-059d9c6119cb
# ╠═66d4e274-f180-11ea-0831-a946570eca63
# ╠═453bb1f2-f18e-11ea-086f-d18ed1055e0f
# ╠═0fa21e04-f192-11ea-03db-7d8b07acfa91
# ╠═a8f5a20a-f195-11ea-167f-7d408ebe0366
# ╠═3b10da96-f18d-11ea-16fa-bf8437f54903
# ╠═cd4c73b6-f17c-11ea-08be-5bf044147749
# ╠═cea59342-f17d-11ea-11c3-5ddc3062b120
# ╠═03c0a192-f198-11ea-3a75-f7b26638fc6a
# ╟─5b9381f2-f17f-11ea-3fb0-c5f5314c3c69
# ╠═0681da22-f17a-11ea-1fca-a7d4004af233
# ╠═5f8782ca-f17a-11ea-082a-a98ea089b700
# ╠═7bea3116-f18d-11ea-19d8-e5ef06dd114a
# ╠═b0d4d10c-f191-11ea-2f84-1b19dcb85d6a
# ╠═16504abe-f180-11ea-33b6-bfb2937f0846
# ╠═07a889aa-f193-11ea-3a0d-1bacdd6cc2cb
# ╠═5a869532-f18d-11ea-2f3e-2b5886e7fa70
# ╠═5cd02706-f181-11ea-217e-c167fc0a4164
# ╠═af3428da-f181-11ea-0ea8-f9bfabc320d3
# ╠═f36eea06-f192-11ea-003e-b91e794f3a6f
