using Revise
using ReversibleSeismic

nx = ny = 100
nstep = 2000
param = AcousticPropagatorParams(nx=nx, ny=ny,
     Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=nstep)

c = 1000*ones(param.NX+2, param.NY+2)
srci = param.NX÷2
srcj = param.NY÷2
srcv = Ricker(param, 100.0, 500.0)

tu = zeros(size(c)..., nstep+1)
tφ = zeros(size(c)..., nstep+1)
tψ = zeros(size(c)..., nstep+1)
i_solve!(param, srci, srcj, srcv, c, tu, tφ, tψ)
loss = sum(tu .^ 2)
@assert loss ≈ 10.931466822080788

using PyPlot 

function viz(param, tu)
     close("all")
     plot(srcv)
     savefig("_srcv.png")

     for i = 1:200:param.NSTEP+1
          close("all")
          pcolormesh(tu[:,:,i])
          colorbar()
          savefig("_f$i.png")
     end
end

viz(param, tu)
