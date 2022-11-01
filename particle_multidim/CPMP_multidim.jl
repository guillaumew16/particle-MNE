module CPMPMultidim

export run_CPMP, avg_iter

import ForwardDiff
using Random
using ProgressMeter
include("../utils/misc.jl")
using .MiscUtils: normalize!

"""
Take a CP gradient step on min_{wx} max_{wy} obj(wx, wy)
- starting from wx,x, wy,y
- evaluating the gradients at wxp,xp, wyp,yp
- with stepsizes eta_wx, eta_x, eta_wy, eta_y
Returns the updated particles wx1,x1, wy1,y1
"""
function step_CPMDA(
        obj,
        wx, x, wy, y,
        wxp, xp, wyp, yp,
        eta_wx, eta_x, eta_wy, eta_y;
        true_prox=false
)
    dimx, mx = size(x)
    dimy, my = size(y)

    wx1 = similar(wx)
    x1 = similar(x)
    wy1 = similar(wy)
    y1 = similar(y)

    Dwx_Objp = ForwardDiff.gradient(wx -> obj(wx, xp, wyp, yp), wxp)
    Dwy_Objp = ForwardDiff.gradient(wy -> obj(wxp, xp, wy, yp), wyp)
    Dx_Objp = ForwardDiff.gradient(x -> obj(wxp, x, wyp, yp), xp) # has the same shape as xp
    Dy_Objp = ForwardDiff.gradient(y -> obj(wxp, xp, wyp, y), yp)
    
    # take step: wx, wy
    wx1 = wx .* exp.(-eta_wx * Dwx_Objp)
    normalize!(wx1)
    wy1 = wy .* exp.(eta_wy * Dwy_Objp)
    normalize!(wy1)

    # take step: x, y
    if true_prox
        for i=1:mx
            x1[:,i] = x[:,i] .- eta_x * Dx_Objp[:,i] ./ wx[i]
        end
        for j=1:my
            y1[:,j] = y[:,j] .+ eta_y * Dy_Objp[:,j] ./ wy[j]
        end
    else
        for i=1:mx
            x1[:,i] = x[:,i] .- eta_x * Dx_Objp[:,i] ./ wxp[i]
        end
        for j=1:my
            y1[:,j] = y[:,j] .+ eta_y * Dy_Objp[:,j] ./ wyp[j]
        end
    end
    
    wx1, x1, wy1, y1
end

"""
Run (min-max) Conic Particle Mirror Prox (CPMP) for T iterations on min_{wx} max_{wy} obj(wx, wy)
    eta0_wx, eta0_x, eta0_wy, eta0_y: initial step-size
        actually we use constant step-size (for now)
    extrasteps (default=2): number of steps to take to approximately solve inner variational problem
        extrasteps=1: CPMDA, extrasteps=2: CPMP
    init_pos: how to initialize the positions
        "iid_unif" to sample iid uniformly on [0,1)
        "grid_unif" to choose a uniform grid on [0,1)
    rng: only used if init_pos == "iid_unif"
    wx00, x00, wy00, y00: if provided, init_pos is ignored and algo is initialized to wx00, x00, wy00, y00
    floorto01: take x and y back to [0,1] at each iteration (for easier plotting for periodic obj)

Returns
    copies_wx: Array mx*(T+1) containing the iterates (t=1 is at initialization)
    copies_x: Array mx*(T+1)
    copies_wy: Array my*(T+1)
    copies_x: Array my*(T+1)
"""
function run_CPMP(obj, dimx, dimy, T, mx, my, eta0_wx, eta0_x, eta0_wy, eta0_y, init_pos; extrasteps=2, rng=MersenneTwister(1234),
        wx00=nothing, x00=nothing, wy00=nothing, y00=nothing,
        true_prox=false)
    ## init
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
    @assert mx==length(wx) && my==length(wy)
    @assert size(x)==(dimx, mx) && size(y)==(dimy, my)

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
        # extrasteps=1: MD-SP, extrasteps=2: MP-SP 
        wxp, xp, wyp, yp = wx, x, wy, y
        for s=1:extrasteps
            wxp, xp, wyp, yp = step_CPMDA(
                obj,
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
