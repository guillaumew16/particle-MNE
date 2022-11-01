module CPMPBilinGameMultidim

export run_CPMP, avg_iter

import ForwardDiff
using Random
using ProgressMeter
include("../utils/misc.jl")
using .MiscUtils: normalize!
include("../utils/multidim.jl")
using .MultidimUtils

"""
Take a step following the particle divergence geometry
- starting from w,z
- along the direction `gdt_w`, `gdt_z` (basically gradient ascent)
- with stepsizes eta_w, eta_z
Returns the updated particles w1,z1
"""
function particle_step_gradient_ascent(w, z, gdt_w, gdt_z, eta_w, eta_z)
    dim, m = size(z)
    w1 = similar(w)
    z1 = similar(z)

    for i=1:m
        w1[i] = w[i] * exp(eta_w * gdt_w[i])
    end
    normalize!(w1)
    for i=1:m
        z1[:,i] = z[:,i] + eta_z * gdt_z[:,i]
    end
    w1, z1
end

"""
Take a CP gradient step
- starting from wx,x, wy,y
- evaluating the gradients at wxp,xp, wyp,yp
- with stepsizes eta_wx, eta_x, eta_wy, eta_y
Returns the updated particles wx1,x1, wy1,y1
"""
function step_CPMDA(
        f,
        wx, x, wy, y,
        wxp, xp, wyp, yp,
        eta_wx, eta_x, eta_wy, eta_y;
        true_prox=false
)
    # f is payoff function (a.k.a gfun, but f is shorter)
    
    dimx, mx = size(x)
    dimy, my = size(y)

    Dxfp = Array{Float64}(undef, dimx, mx, my)
    Dyfp = Array{Float64}(undef, dimy, mx, my)
    for i=1:mx, j=1:my
        Dxfp[:,i,j] = ForwardDiff.gradient(xx -> f(xx,yp[:,j]), xp[:,i])
        Dyfp[:,i,j] = ForwardDiff.gradient(yy -> f(xp[:,i],yy), yp[:,j])
    end

    # gdt_wx = Array{Float64}(undef, mx)
    # for i=1:mx
    #     gdt_wx[i] = sum( wyp[j]*f(xp[:,i],yp[:,j]) for j=1:my )
    # end
    # gdt_x = Array{Float64}(undef, dimx, mx)
    # for i=1:mx
    #     gdt_x[:,i] = sum( wyp[j] * Dxfp[:,i,j] for j=1:my )
    # end
    # wx1, x1 = particle_step_gradient_ascent(wx, x, -gdt_wx, -gdt_x, eta_wx, eta_x)
    
    # gdt_wy = Array{Float64}(undef, my)
    # for j=1:my
    #     gdt_wy[j] = sum( wxp[i]*f(xp[:,i],yp[:,j]) for i=1:mx )
    # end
    # gdt_y = Array{Float64}(undef, dimy, my)
    # for j=1:my
    #     gdt_y[:,j] = sum( wxp[i] * Dxfp[:,i,j] for i=1:mx )
    # end
    # wy1, y1 = particle_step_gradient_ascent(wy, y, gdt_wy, gdt_y, eta_wy, eta_y)
    
    # equivalently:

    # take step: x
    wx1 = similar(wx)
    x1 = similar(x)
    for i=1:mx
        s = sum( wyp[j]*f(xp[:,i],yp[:,j]) for j=1:my )
        wx1[i] = wx[i] * exp(-eta_wx*s)
    end
    normalize!(wx1)
    for i=1:mx
        s = sum( wyp[j] * Dxfp[:,i,j] for j=1:my )
        @assert size(s) == (dimx,)
        if true_prox
            x1[:,i] = x[:,i] - eta_x * wxp[i] / wx[i] * s
        else
            x1[:,i] = x[:,i] - eta_x * s
        end
    end
    
    # take step: y
    wy1 = similar(wy)
    y1 = similar(y)
    for j=1:my
        s = sum( wxp[i]*f(xp[:,i],yp[:,j]) for i=1:mx )
        wy1[j] = wy[j] * exp(eta_wy*s)
    end
    normalize!(wy1)
    for j=1:my
        s = sum( wxp[i] * Dyfp[:,i,j] for i=1:mx )
        @assert size(s) == (dimy,)
        if true_prox
            y1[:,j] = y[:,j] + eta_y * wyp[j] / wy[j] * s
        else
            y1[:,j] = y[:,j] + eta_y * s
        end
    end
    
    wx1, x1, wy1, y1
end

"Equivalent to step_CPMDA, might be marginally faster in some cases (?)"
function my_step_CPMDA(
        f,
        wx, x, wy, y,
        wxp, xp, wyp, yp,
        eta_wx, eta_x, eta_wy, eta_y;
        true_prox=false
)
    # f is payoff function (a.k.a gfun, but f is shorter)
    dimx, mx = size(x)
    dimy, my = size(y)
    fp = Array{Float64}(undef, mx, my)
    for j=1:my, i=1:mx
        @views fp[i,j] = f(xp[:,i],yp[:,j])
    end
    Dxfp = Array{Float64}(undef, dimx, mx, my)
    Dyfp = Array{Float64}(undef, dimy, mx, my)
    for j=1:my, i=1:mx
        @views Dxfp[:,i,j] .= ForwardDiff.gradient(xx -> f(xx,yp[:,j]), xp[:,i])
        @views Dyfp[:,i,j] .= ForwardDiff.gradient(yy -> f(xp[:,i],yy), yp[:,j])
    end
    # take step: x
    wx1 = wx .* exp.(.-eta_wx .* (fp * wyp))
    LinearAlgebra.normalize!(wx1, 1)
    s = Array{Float64}(undef, dimx, mx)
    @views for i=1:mx
        mul!(s[:,i], Dxfp[:,i,:], wyp)
    end
    if true_prox
        x1 = x .- eta_x .* wxp ./ wx .* s
    else
        x1 = x .- eta_x .* s
    end
    # take step: y
    wy1 = wy .* exp.(eta_wy .* (transpose(fp) * wxp))
    LinearAlgebra.normalize!(wy1, 1)
    s = Array{Float64}(undef, dimy, my)
    @views for j=1:my
        mul!(s[:,j], Dyfp[:,:,j], wxp)
    end
    if true_prox
        y1 = y .+ eta_y .* wyp ./ wy .* s
    else
        y1 = y .+ eta_y .* s
    end
    return wx1, x1, wy1, y1
end

"""
Run (min-max) Conic Particle Mirror Prox (CPMP) for T iterations
    eta0_wx, eta0_x, eta0_wy, eta0_y: initial step-size
        actually we use constant step-size (for now)
    extrasteps (default=2): number of steps to take to approximately solve inner variational problem
        extrasteps=1: CPMDA, extrasteps=2: CPMP
    init_pos: how to initialize the positions
        "iid_unif" to sample iid uniformly on [0,1)^d
        "grid_unif" to choose a uniform grid on [0,1)^d
    rng: only used if init_pos == "iid_unif"

Returns
    copies_wx: Array mx*(T+1) containing the iterates (t=1 is at initialization)
    copies_x: Array mx*(T+1)
    copies_wy: Array my*(T+1)
    copies_x: Array my*(T+1)
"""
function run_CPMP(f, dimx, dimy, T, mx, my, eta0_wx, eta0_x, eta0_wy, eta0_y, init_pos; extrasteps=2, rng=MersenneTwister(1234),
        wx00=nothing, x00=nothing, wy00=nothing, y00=nothing,
        true_prox=false)
    # f is payoff function (a.k.a gfun, but f is shorter)

    ## init
    # if !isnothing(wx00) && !isnothing(x00) && !isnothing(wy00) && !isnothing(y00)
    #     wx = wx00
    #     x = x00
    #     wy = wy00
    #     y = y00
    # else
    wx = ones(mx) ; normalize!(wx)
    wy = ones(my) ; normalize!(wy)
    if init_pos == "iid_unif"
        x = rand(rng, dimx, mx) # iid uniformly distributed in [0,1)^dimx
        y = rand(rng, dimy, my)
    elseif init_pos == "grid_unif"
        x, mxx = initialize_positions_grid_unif(dimx, mx)
        if mxx != mx
            # open(logfile, "a") do f
            #     write(f, "Using grid_unif initialization with mx=$(mxx)=$(mxx^(1/dimx))^$(dimx) instead of $(mx) particles. (dimx=$(dimx))\n")
            # end
            @info "Using grid_unif initialization with mx=$(mxx)=$(mxx^(1/dimx))^$(dimx) instead of $(mx) particles. (dimx=$(dimx))"
            mx = mxx
        end
        y, myy = initialize_positions_grid_unif(dimy, my)
        if myy != my
            # open(logfile, "a") do f
            #     write(f, "Using grid_unif initialization with my=$(myy)=$(myy^(1/dimy))^$(dimy) instead of $(my) particles. (dimy=$(dimy))\n")
            # end
            @info "Using grid_unif initialization with my=$(myy)=$(myy^(1/dimy))^$(dimy) instead of $(my) particles. (dimy=$(dimy))"
            my = myy
        end
    end
    wx = isnothing(wx00) ? wx : wx00
    x = isnothing(x00) ? x : x00
    wy = isnothing(wy00) ? wy : wy00
    y = isnothing(y00) ? y : y00

    # for extragradient step
    wxp = similar(wx)
    wyp = similar(wy)
    xp = similar(x)
    yp = similar(y)

    # to store intermediate values
    copies_wx = Array{Float64}(undef, mx, T+1)
    copies_wy = Array{Float64}(undef, my, T+1)
    copies_x = Array{Float64}(undef, dimx, mx, T+1)
    copies_y = Array{Float64}(undef, dimy, my, T+1)

    eta_wx, eta_x, eta_wy, eta_y = eta0_wx, eta0_x, eta0_wy, eta0_y # initialize stepsizes

    ## loop
    @showprogress 1 for t=1:T # minimum update interval of 1 second
    # for t=1:T
        copies_wx[:,t] = copy(wx)
        copies_wy[:,t] = copy(wy)
        copies_x[:,:,t] = copy(x)
        copies_y[:,:,t] = copy(y)

        # extragradient ("ghost" step)
        wxp, xp, wyp, yp = wx, x, wy, y
        for s=1:extrasteps
            wxp, xp, wyp, yp = step_CPMDA(
                f,
                wx, x, wy, y,
                wxp, xp, wyp, yp,
                eta_wx, eta_x, eta_wy, eta_y;
                true_prox=true_prox
            )
        end

        # take step
        wx, x, wy, y = wxp, xp, wyp, yp
    end

    copies_wx[:,T+1] = copy(wx)
    copies_wy[:,T+1] = copy(wy)
    copies_x[:,:,T+1] = copy(x)
    copies_y[:,:,T+1] = copy(y)
    
    return copies_wx, copies_x, copies_wy, copies_y
end

function avg_iter(copies_wx, copies_x, copies_wy, copies_y)
    T = size(copies_wx)[2] - 1
    dimx, mx, _ = size(copies_x)
    dimy, my, _ = size(copies_y)
    avg_wx = vec(copies_wx) / (T+1)
    avg_wy = vec(copies_wy) / (T+1)
    avg_x = reshape(copies_x, dimx, mx*(T+1))
    avg_y = reshape(copies_y, dimy, my*(T+1))
    # sum(avg_wx), sum(avg_wy)
    return avg_wx, avg_x, avg_wy, avg_y
end

end
