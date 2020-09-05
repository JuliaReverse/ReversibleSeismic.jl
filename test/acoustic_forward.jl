using Revise
using ReversibleSeismic

param = AcousticPropagatorParams(NX = 100, NY = 100,
     Rcoef=0.2, DELTAX=20, DELTAY=20, DELTAT=0.05, NSTEP=20,
     USE_PML_XMIN = true, USE_PML_XMAX = true, 
     USE_PML_YMIN = true, USE_PML_YMAX = true) 

c = 1000*ones(param.NX+2, param.NY+2)
srci = param.NX÷2
srcj = param.NY÷2
srcv = Ricker(param, 100.0, 500.0)
tu = AcousticPropagatorSolver(param, srci, srcj, srcv, c)
loss = sum(tu .^ 2)
@assert loss ≈ 10.931466822080788

#=
using PyPlot 

function viz(param， tu)
     close("all")
     plot(srcv)
     savefig("_srcv.png")

     for i = 1:200:param.NSTEP+1
          close("all")
          pcolormesh(reshape(tu[:,i], param.NX+2, param.NY+2))
          colorbar()
          savefig("_f$i.png")
     end
end

viz(tu, param)
=#
