using Revise
using ReversibleSeismic

param = AcousticPropagatorParams(NX = 100, NY = 100,
     Rcoef=0.2, DELTAX=20, DELTAY=20, DELTAT=0.01, NSTEP=100) 

c = ones(param.NX+2, param.NY+2)
srci = param.NX÷2
srcj = param.NY÷2
srcv = Ricker(param, 100.0, 500.0)
tu = AcousticPropagatorSolver(param, srci, srcj, srcv, c)