module CPMPBilinGame1D

export run_CPMP, avg_iter

import ForwardDiff
using Random
using ProgressMeter
include("../utils/misc.jl")
using .MiscUtils: normalize!

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
    
    mx, my = length(wx), length(wy)
    wx1 = similar(wx)
    x1 = similar(x)
    wy1 = similar(wy)
    y1 = similar(y)

    Dxfp = Array{Float64}(undef, mx, my)
    Dyfp = Array{Float64}(undef, mx, my)
    for i=1:mx, j=1:my
        Dxfp[i,j] = ForwardDiff.derivative(xx -> f(xx,yp[j]), xp[i])
        Dyfp[i,j] = ForwardDiff.derivative(yy -> f(xp[i],yy), yp[j])
    end
    
    # take step: x
    for i=1:mx
        s = sum( wyp[j]*f(xp[i],yp[j]) for j=1:my )
        wx1[i] = wx[i] * exp(-eta_wx*s)
    end
    normalize!(wx1)
    for i=1:mx
        s = sum( wyp[j] * Dxfp[i,j] for j=1:my )
        if true_prox
            x1[i] = x[i] - eta_x * wxp[i] / wx[i] * s
        else
            x1[i] = x[i] - eta_x * s
        end
    end
    
    # take step: y
    for j=1:my
        s = sum( wxp[i]*f(xp[i],yp[j]) for i=1:mx )
        wy1[j] = wy[j] * exp(eta_wy*s)
    end
    normalize!(wy1)
    for j=1:my
        s = sum( wxp[i] * Dyfp[i,j] for i=1:mx )
        if true_prox
            y1[j] = y[j] + eta_y * wyp[j] / wy[j] * s
        else
            y1[j] = y[j] + eta_y * s
        end
    end
    
    wx1, x1, wy1, y1
end

"""
Run (min-max) Conic Particle Mirror Prox (CPMP) for T iterations
    eta0_wx, eta0_x, eta0_wy, eta0_y: initial step-size
        actually we use constant step-size (for now)
    extrasteps (default=2): number of steps to take to approximately solve inner variational problem
        extrasteps=1: CPMDA, extrasteps=2: CPMP
    init_pos: how to initialize the positions
        "iid_unif" to sample iid uniformly on [0,1)
        "grid_unif" to choose a uniform grid on [0,1)
    rng: only used if init_pos == "iid_unif"
    wx00, x00, wy00, y00: if provided, init_pos is ignored and algo is initialized to wx00, x00, wy00, y00
    floorto01: take x and y back to [0,1] at each iteration (for easier plotting for periodic f)

Returns
    copies_wx: Array mx*(T+1) containing the iterates (t=1 is at initialization)
    copies_x: Array mx*(T+1)
    copies_wy: Array my*(T+1)
    copies_x: Array my*(T+1)
"""
function run_CPMP(f, T, mx, my, eta0_wx, eta0_x, eta0_wy, eta0_y, init_pos; extrasteps=2, rng=MersenneTwister(1234),
        wx00=nothing, x00=nothing, wy00=nothing, y00=nothing,
        true_prox=false,
        floorto01=true,
        xinitmin=0., xinitmax=1., yinitmin=0., yinitmax=1.)
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
        x = xinitmin .+ (xinitmax-xinitmin) .* rand(rng, mx) # iid uniformly distributed in [xinitmin,xinitmax)
        y = yinitmin .+ (yinitmax-yinitmin) .* rand(rng, my)
    elseif init_pos == "grid_unif"
        x = collect(range(xinitmin, stop=xinitmax, length=mx+1)[1:mx])
        y = collect(range(yinitmin, stop=yinitmax, length=my+1)[1:my])
    end
    wx = isnothing(wx00) ? wx : wx00
    x = isnothing(x00) ? x : x00
    wy = isnothing(wy00) ? wy : wy00
    y = isnothing(y00) ? y : y00
    @assert mx==length(wx) && mx==length(x) && my==length(wy) && my==length(y)

    # for extragradient step
    wxp = similar(wx)
    wyp = similar(wy)
    xp = similar(x)
    yp = similar(y)

    # to store intermediate values
    copies_wx = Array{Float64}(undef, mx, T+1)
    copies_wy = Array{Float64}(undef, my, T+1)
    copies_x = Array{Float64}(undef, mx, T+1)
    copies_y = Array{Float64}(undef, my, T+1)

    eta_wx, eta_x, eta_wy, eta_y = eta0_wx, eta0_x, eta0_wy, eta0_y # initialize stepsizes

    ## loop
    @showprogress 1 for t=1:T # minimum update interval of 1 second
    # for t=1:T
        copies_wx[:,t] = copy(wx)
        copies_wy[:,t] = copy(wy)
        copies_x[:,t] = copy(x)
        copies_y[:,t] = copy(y)

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

        # take x and y back to [0,1) (equivalent since f periodic but easier to plot)
        if floorto01
            x = x .- floor.(x)
            y = y .- floor.(y)
        end
    end

    copies_wx[:,T+1] = copy(wx)
    copies_wy[:,T+1] = copy(wy)
    copies_x[:,T+1] = copy(x)
    copies_y[:,T+1] = copy(y)
    
    return copies_wx, copies_x, copies_wy, copies_y
end

function avg_iter(copies_wx, copies_x, copies_wy, copies_y)
    T = size(copies_wx)[2] - 1
    avg_wx = vec(copies_wx) / (T+1)
    avg_wy = vec(copies_wy) / (T+1)
    avg_x = vec(copies_x)
    avg_y = vec(copies_y)
    # sum(avg_wx), sum(avg_wy)
    return avg_wx, avg_x, avg_wy, avg_y
end

end
