

function visualize_wavefield(val::Array{Float64, 3}, 
    param::Union{ElasticPropagatorParams,AcousticPropagatorParams, MPIAcousticPropagatorParams, MPIElasticPropagatorParams}; 
    dirs::String="figure", kwargs...) 
    
    if !isdir(dirs)
        dirs="./"
    end
    
    figure()
    vmin = -3std(val)
    vmax = 3std(val)
    dt = max(1, div(param.NSTEP, 20))

    pl = matshow(val[1,:,:]', cmap="jet", vmin=vmin, vmax=vmax)
    t = title("Time = $(round(0.0, digits=2))")
    colorbar(shrink=0.8)
    gca().xaxis.set_ticks_position("bottom")
    xlabel("X")
    ylabel("Y")

    function update(i)
        pl[:set_data](val[i,:,:]')
        t.set_text("Time = $(round((i-1)*param.DELTAT, digits=2))")
    end
    p = animate(update, [2:dt:size(val, 1);])
    saveanim(p, joinpath(dirs, "forward_wavefield.gif"))

    return p
end

function plot_result(sess, var, feed_dict, iter; figure_dir::String="figures", result_dir::String="results", var_name=nothing)

    if !isdir(figure_dir)
        figure_dir="./"
    end
    if !isdir(result_dir)
        result_dir=figure_dir
    end

    x = run(sess, var, feed_dict=feed_dict)

    std_images = zeros(size(x,2), size(x,3)) * 0
    mean_images = zeros(size(x,2), size(x,3)) * 0
    for i = 1:size(x,2)
        for j = 1:size(x, 3)
            std_images[i,j] = std(x[:, i, j, 1])
            mean_images[i,j] = mean(x[:, i, j, 1])
        end
    end

    figure()
    imshow(std_images')
    colorbar(shrink=0.5)
    title("Iteration = $iter")
    if isnothing(var_name)
        savefig(joinpath(figure_dir, "std_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "std_$(lpad(iter,5,"0")).txt"), std_images')
    else
        savefig(joinpath(figure_dir, "std_$(var_name)_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "std_$(var_name)_$(lpad(iter,5,"0")).txt"), std_images')
    end

    figure()
    fig, ax = subplots(div(size(x)[1],2), 2)
    for i = 1:div(size(x)[1],2)
        for j = 1:2
            ax[i, j].get_xaxis().set_visible(false)
            ax[i, j].get_yaxis().set_visible(false)
            k = (i-1) * 2 + j
            pcm= ax[i, j].imshow(x[k,:,:]')
            # fig.colorbar(pcm, ax=ax[i,j])
        end
    end
    subplots_adjust(wspace=0, hspace=0)
    suptitle("Iteration = $iter")
    if isnothing(var_name)
        savefig(joinpath(figure_dir, "inv_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), mean_images')
    else
        savefig(joinpath(figure_dir, "inv_$(var_name)_$(lpad(iter,5,"0")).png"), bbox_inches="tight")
        writedlm(joinpath(result_dir, "inv_$(var_name)_$(lpad(iter,5,"0")).txt"), mean_images')
    end
    # writedlm(joinpath(result_dir, "inv_$(lpad(iter,5,"0")).txt"), vp*scale)
    close("all")
end

function visualize_model(vp::Array{Float64, 2}, params::Union{ElasticPropagatorParams,AcousticPropagatorParams, MPIAcousticPropagatorParams})
    clf()
    if isa(params, MPIAcousticPropagatorParams)
        pcolormesh([0:params.NX-1;]*params.DELTAX/1e3,[0:params.NY-1;]*params.DELTAY/1e3,  Array(vp'))
    else
        pcolormesh([0:params.NX+1;]*params.DELTAX/1e3,[0:params.NY+1;]*params.DELTAY/1e3,  Array(vp'))
    end
    axis("scaled")
    colorbar(shrink=0.4)
    xlabel("x (km)")
    ylabel("z (km)")
    gca().invert_yaxis()
end

export Ricker
"""
    Ricker(epp::Union{ElasticPropagatorParams, AcousticPropagatorParams}, 
    a::Union{PyObject, <:Real}, 
    shift::Union{PyObject, <:Real}, 
    amp::Union{PyObject, <:Real}=1.0)

Returns a Ricker wavelet (a tensor). 
- `epp`: a `ElasticPropagatorParams` or an `AcousticPropagatorParams`
- `a`: Width parameter
- `shift`: Center of the Ricker wavelet
- `amp`: Amplitude of the Ricker wavelet

```math
f(x) = \\mathrm{amp}A (1 - x^2/a^2) exp(-x^2/2 a^2)
```
where 
```math
A = 2/sqrt(3a)pi^1/4
```
"""
function Ricker(epp, 
        a, 
        shift, 
        amp=1.0)
    a = convert_to_tensor(a)
    shift = convert_to_tensor(shift)
    amp = convert_to_tensor(amp)
    NT, T = epp.NSTEP, epp.NSTEP*epp.DELTAT
    A = 2 / (sqrt(3 * a) * (pi^0.25))
    wsq = a^2
    vec = collect(1:NT)-shift
    xsq = vec^2
    mod = (1 - xsq / wsq)
    gauss = exp(-xsq / (2 * wsq))
    total = amp * A * mod * gauss
    return total
end
