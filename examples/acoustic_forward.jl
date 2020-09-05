using Revise
using ReversibleSeismic

param = AcousticPropagatorParams(nx = 100, ny = 100,
     Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep=2000)

c = 1000*ones(param.NX+2, param.NY+2)
srci = param.NX÷2
srcj = param.NY÷2
srcv = Ricker(param, 100.0, 500.0)
tu = solve(param, srci, srcj, srcv, c)
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
