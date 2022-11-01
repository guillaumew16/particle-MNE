module PlotGame1DUtils

export heatmap_gfun_1D, contour_gfun_1D,
    plot_first_variations_1D,
    plot_iter_1D,
    plot_iter_smooth_1D,
    smoothen_iters

import DSP
using Plots
pyplot()

include("../utils/misc.jl")
using .MiscUtils
include("game_theory_contdom_1D.jl")
using .GameTheoryContdom1DUtils: locmat

function heatmap_gfun_1D(gfun; len_xs=401, len_ys=401, xmin=0., xmax=1., ymin=0., ymax=1.)
    xs = range(xmin, stop=xmax, length=len_xs)
    ys = range(ymin, stop=ymax, length=len_ys)
    gfuns = locmat(gfun, xs, ys)
    heatmap(transpose(gfuns))
end

function contour_gfun_1D(gfun; len_xs=401, len_ys=401, xmin=0., xmax=1., ymin=0., ymax=1.)
    xs = range(xmin, stop=xmax, length=len_xs)
    ys = range(ymin, stop=ymax, length=len_ys)
    contour(xs, ys, gfun, fill=true, aspect_ratio=1)
end

"""
(Continuous domain) Plot the first variations w.r.t x and y at reference point (wx0, x0, wy0, y0)
At optimum, we know that
- First variation w.r.t mu, (F nu)(x), is >=gval everywhere and =gval for x in support(mu)
- First variation w.r.t nu, (mu^T F)(y), is <=gval everywhere and =gval for y in support(nu)
Returns plt_firstvar_x, plt_firstvar_y
"""
function plot_first_variations_1D(gfun, wx0, x0, wy0, y0, gval0; 
        hidetitle=false, hidelabels=false, hidestems=false,
        len_xs=401, len_ys=401, xmin=0., xmax=1., ymin=0., ymax=1.)
    xs = range(xmin, stop=xmax, length=len_xs)
    ys = range(ymin, stop=ymax, length=len_ys)
    Fnu0_s = locmat(gfun, xs, y0) * wy0 # approximate continuous first-variation function w.r.t x
    mu0F_s = transpose(wx0) * locmat(gfun, x0, ys) # approximate continuous first-variation function w.r.t y
    mu0F_s = mu0F_s[:] # convert (*×1) Matrix to Vector
    
    plt_firstvar_x = plot(xs, Fnu0_s, xlabel="x", label="")
    hline!([gval0], label=(hidelabels ? "" : "payoff"))
    if !hidestems
        vline!([x0], marker_z=wx0, color=:greens, colorbar_entry=false, label=(hidelabels ? "" : "x at ref point"))
    end
    if !hidetitle
        title!("First variation w.r.t wx at reference point")
    end
    # it should be >=payoff everywhere and =payoff for i where wx[i] > 0
    
    plt_firstvar_y = plot(ys, mu0F_s, xlabel="y", label="") #, legend_position=:bottomright)
    hline!([gval0], label=(hidelabels ? "" : "payoff"))
    if !hidestems
        vline!([y0], marker_z=wy0, color=:greens, colorbar_entry=false, label=(hidelabels ? "" : "y at ref point"))
    end
    if !hidetitle
        title!("First variation w.r.t wy at reference point")
    end
    # it should be <=payoff everywhere and =payoff for j where wy[j] > 0

    return plt_firstvar_x, plt_firstvar_y, Fnu0_s, mu0F_s
end

"Self-contained function to plot one iteration only"
function plot_iter_1D(gfun, wx, x, wy, y;
        figsize=(600, 600), hidelabels=false, showpayoff="heatmap",
        len_xs=401, len_ys=401, xmin=0., xmax=1., ymin=0., ymax=1.)
    xs = range(xmin, stop=xmax, length=len_xs)
    ys = range(ymin, stop=ymax, length=len_ys)
    # gfuns = locmat(gfun, xs, ys)

    x01 = linrescale(x, xmin, xmax)
    y01 = linrescale(y, ymin, ymax)
    smoothsig_wx = smoothen_iter(wx, x01; len_s=len_xs)
    smoothsig_wy = smoothen_iter(wy, y01; len_s=len_ys)
    
    pltt = plot_iter_smooth_1D(wx, x, wy, y, 
        smoothsig_wx, smoothsig_wy;
        # gfuns=gfuns,
        gfun=gfun,
        showpayoff=showpayoff, showlines=true, figsize=figsize, hidelabels=hidelabels,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    return pltt, smoothsig_wx, smoothsig_wy
end

"""
Plot an iteration given smoothened version of it.
This function is designed to allow efficiently plotting many iterations:
- providing the matrix gfuns avoids recomputing across calls (only one of gfun or gfuns should be provided!)
- smoothsig_wx, smoothsig_wy can be computed for several iterates at once by calling smoothen_iters(copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y)

Layout adapted from pythonot 0.7.0: https://pythonot.github.io/_modules/ot/plot.html#plot1D_mat
"""
function plot_iter_smooth_1D(wx, x, wy, y, smoothsig_wx, smoothsig_wy;
        gfun=nothing, gfuns=nothing,
        showpayoff="heatmap", showlines=false, figsize=(600, 600), hidelabels=false,
        xmin=0., xmax=1., ymin=0., ymax=1.)
    len_xs = length(smoothsig_wx)
    len_ys = length(smoothsig_wy)
    xs = range(xmin, stop=xmax, length=len_xs)
    ys = range(ymin, stop=ymax, length=len_ys)
    
    x01 = linrescale(x, xmin, xmax)
    y01 = linrescale(y, ymin, ymax)

    # optionally ask user to provide gfuns to avoid recomputing
    @assert xor(isnothing(gfun), isnothing(gfuns))
    if isnothing(gfuns)
        gfuns = locmat(gfun, xs, ys)
    else
        @assert size(gfuns) == (len_xs, len_ys)
    end
    
    l = @layout [_ axx{0.7w,0.3h}; axy{0.3w,0.7h} axf]
    axx = plot(xs, smoothsig_wx, 
        label=(hidelabels ? false : "wx"),
        xlabel="x",
        xlim=(xmin, xmax),
        yticks=false,
        ymirror=true, xmirror=true)
    axy = plot(smoothsig_wy, ys, 
        label=(hidelabels ? false : "wy"), legend_position=:topleft,
        ylabel="y",
        ylim=(ymin, ymax),
        xticks=false,
        xflip=true, yflip=false)
    if showpayoff == "contour"
        @assert !isnothing(gfun)
        axf = contour(xs, ys, gfun, fill=true, aspect_ratio=1, colorbar_entry=false, showaxis=false)
        # add things to the contour subplot
        if showlines
            vline!(x01, marker_z=wx,
                # alpha=wx,
                color=:greens, label="", colorbar_entry=false)
            hline!(y01, marker_z=wy,
                # alpha=wy,
                color=:greens, label="", colorbar_entry=false)
        end
    elseif showpayoff == "heatmap"
        axf = heatmap(transpose(gfuns),
            # aspect_ratio=1,
            colorbar_entry=false, showaxis=false)
        # add things to the contour subplot
        if showlines
            vline!(len_xs .* x01, marker_z=wx,
                # alpha=wx,
                color=:greens, label="", colorbar_entry=false)
            hline!(len_ys .* y01, marker_z=wy,
                # alpha=wy,
                color=:greens, label="", colorbar_entry=false)
        end
    else
        throw(ArgumentError("showpayoff must be \"heatmap\" or \"contour\""))
    end

    return plot(axx, axy, axf, layout=l, size=figsize)
end


####################
## Utility functions

"""
Interpret a sum-of-dirac distribution wx,x as a signal over its domain [0,1) discretized with step 1/len_xs
Returns sig_wx: Vector of length len_xs
NB: Assumes the domain is [0,1)
"""
function particles2signal(wx, x, len_xs=401)
    sig_wx = zeros(len_xs)
    pos2idx(pos, len) = ( @assert 0<=pos && pos<1 ; Int(floor(len*pos)) % len + 1)
    for i in eachindex(x)
        sig_wx[pos2idx(x[i], len_xs)] += wx[i]
    end
    return sig_wx
end
"""
Returns a smoothened normalized version of a 1d signal
- sig: 1d signal over [0,1)
- delta: window size
Must have delta > 1/signallength.
(Could also use different window functions, see https://docs.juliadsp.org/stable/windows/)
NB: Assumes the domain is [0,1)
"""
function smoothen(sig, delta)
    length = size(sig)[1]
    window = DSP.Windows.rect(Int(floor(delta*length)))
    out = DSP.filt(window, sig)
    normalize!(out)
    return out
end
"""
Returns smoothened normalized versions of 1d signals
- sigs: Array of 1d signals over [0,1). First dimension represents the variable
- delta: window size
Must have delta > 1/signallength.
(Could also use different window functions, see https://docs.juliadsp.org/stable/windows/)
NB: Assumes the domain is [0,1)
"""
function smoothen_multi(sigs, delta)
    length = size(sigs)[1]
    window = DSP.Windows.rect(Int(floor(delta*length)))
    out = DSP.filt(window, sigs) # DSP.filt applies filter along first dimension
    for k in CartesianIndices(out[1,:])
        normalize!(out[:,k])
    end
    return out
end

# ## small test/illustration
# z = zeros(50)
# z[24] = 1
# smoothz = smoothen(z, 1e-1)
# plot([z, smoothz])

"""
Smoothened version of the distribution w, z
Returns smoothsig_w
NB: Assumes the domain is [0,1)
"""
function smoothen_iter(w, z; len_s=401, smoothdelta=1e-2)
    sig_w = particles2signal(w, z, len_s)
    smoothsig_w = smoothen(sig_w, smoothdelta)
    return smoothsig_w
end
"""
Smoothened versions of the distributions wx,x, wy,y
Returns copies_smoothsig_wx, copies_smoothsig_wy, avg_smoothsig_wx, avg_smoothsig_wy
NB: Assumes the domain is [0,1)
"""
function smoothen_iters(copies_wx, copies_x, copies_wy, copies_y, 
        avg_wx, avg_x, avg_wy, avg_y; 
        len_xs=401, len_ys=401, smoothdeltax=1e-2, smoothdeltay=1e-2)
    T = size(copies_wx)[2] - 1

    copies_sig_wx = zeros(len_xs, T+1)
    copies_sig_wy = zeros(len_ys, T+1)
    for t=1:(T+1)
        copies_sig_wx[:,t] = particles2signal(copies_wx[:,t], copies_x[:,t], len_xs)
        copies_sig_wy[:,t] = particles2signal(copies_wy[:,t], copies_y[:,t], len_ys)
    end
    copies_smoothsig_wx = smoothen_multi(copies_sig_wx, smoothdeltax)
    copies_smoothsig_wy = smoothen_multi(copies_sig_wy, smoothdeltay)

    avg_sig_wx = particles2signal(avg_wx, avg_x, len_xs)
    avg_sig_wy = particles2signal(avg_wy, avg_y, len_ys)
    avg_smoothsig_wx = smoothen(avg_sig_wx, smoothdeltax)
    avg_smoothsig_wy = smoothen(avg_sig_wy, smoothdeltay)
    
    return copies_smoothsig_wx, copies_smoothsig_wy, avg_smoothsig_wx, avg_smoothsig_wy
end

end
