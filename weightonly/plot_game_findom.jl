module PlotGameFindomUtils

export heatmap_gmat, 
    plot_iter_findom, 
    plot_first_variations_findom

using Plots
pyplot()

function heatmap_gmat(gmat)
    return heatmap(transpose(gmat))
end

function plot_iter_findom(gmat, wx, wy; figsize=(600, 600), hidelabels=false)
    # adapted from pythonot 0.7.0: https://pythonot.github.io/_modules/ot/plot.html#plot1D_mat
    mx, my = size(gmat)

    l = @layout [_ axx{0.7w,0.3h}; axy{0.3w,0.7h} axf]
    axx = bar(wx, 
        label=(hidelabels ? false : "wx"),
        xlim=(0.5,mx+0.5),
        yticks=false,
        ymirror=true, xmirror=true)
    axy = bar(wy, 
        label=(hidelabels ? false : "wy"), orientation=:h, legend_position=:topleft,
        ylim=(0.5,my+0.5),
        xticks=false,
        xflip=true, yflip=false)
    axf = heatmap(transpose(gmat), colorbar_entry=false, showaxis=false)
    
    return plot(axx, axy, axf, layout=l, size=figsize)
end

"""
(Finite domain) Plot the first variations w.r.t x and y at reference point (wx0, wy0)
At optimum we know that
- First variation w.r.t a, (M b)_i, is >=gval everywhere and =gval for i s.t a_i>0 (wx[i]>0)
- First variation w.r.t b, (M' a)_j, is <=gval everywhere and =gval for j s.t b_j>0 (wy[j]>0)
Returns plt_firstvar_x, plt_firstvar_y
"""
function plot_first_variations_findom(gmat, wx0, wy0, gval0)
    mx, my = size(gmat)
    
    plt_firstvar_x = plot(range(0, stop=1, step=1/mx), gmat*wy0, xlabel="x", label="first variation")
    hline!([gval0], label="payoff")
    title!("First variation w.r.t wx at reference point")
    # it should be >=payoff everywhere and =payoff for i where wx[i] > 0

    plt_firstvar_y = plot(range(0, stop=1, step=1/my), transpose(gmat)*wx0, xlabel="y", label="first variation")
    hline!([gval0], label="payoff")
    title!("First variation w.r.t wy at reference point")
    # it should be <=payoff everywhere and =payoff for j where wy[j] > 0

    return plt_firstvar_x, plt_firstvar_y
end


end
