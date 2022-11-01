module PlotResFindom

export plot_NI_err, plot_div_to_ref, 
    plot_iters, plot_firstvars

using ProgressMeter
using Plots
pyplot()

include("MP_bilin_game.jl")
using .MPBilinGame
include("plot_game_findom.jl")
using .PlotGameFindomUtils
include("game_theory_findom.jl")
using .GameTheoryFindomUtils

"Compute and plot NI error of iterates"
function plot_NI_err(gmat, T, copies_wx, copies_wy, avg_wx, avg_wy,
        resultdir, logfile, 
        evalNIevery=1, skip_avg=false)

    nierrs = Array{Float64}(undef, T+2)
    for t=1:evalNIevery:(T+1)
        nierrs[t] = NI_err(gmat, copies_wx[:,t], copies_wy[:,t])
    end
    nierrs[T+2] = NI_err(gmat, avg_wx, avg_wy)
    open(logfile, "a") do f
        for t=1:evalNIevery:(T+1)
            write(f, "NI_err at iteration#$t: $(nierrs[t])\n")
        end
        write(f, "NI_err at avg iterate: $(nierrs[T+2])\n")
    end

    plt_NI = plot(range(1, stop=T+1, step=evalNIevery), nierrs[1:evalNIevery:T+1], xlabel="k", label="")
    if !skip_avg
        hline!([nierrs[T+2]], label="avg iterate")
    end
    title!("NI error of iterates")
    fn = "$resultdir/NI_errors.png"
    savefig(plt_NI, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    plt_NI_log = plot(range(1, stop=T+1, step=evalNIevery), eps .+ max.(0, nierrs[1:evalNIevery:T+1]), xlabel="k", yscale=:log10, label="")
    if !skip_avg
        hline!([eps + max(0, nierrs[T+2])], label="avg iterate")
    end
    title!("NI error of iterates (log-linear scale)")
    fn = "$resultdir/NI_errors_logscale.png"
    savefig(plt_NI_log, fn)
    
    plt_NI_log_log = plot(range(1, stop=T+1, step=evalNIevery), eps .+ max.(0, nierrs[1:evalNIevery:T+1]), xlabel="k", yscale=:log10, xscale=:log10, label="")
    if !skip_avg
        hline!([eps + max(0, nierrs[T+2])], label="avg iterate")
    end
    title!("NI error of iterates (log-log scale)")
    fn = "$resultdir/NI_errors_loglogscale.png"
    savefig(plt_NI_log_log, fn)

    return plt_NI, plt_NI_log, plt_NI_log_log, nierrs
end

"Compute and plot KL-divergence to reference point of iterates"
function plot_div_to_ref(gmat, T, copies_wx, copies_wy, avg_wx, avg_wy,
        wx0, wy0,
        resultdir, logfile, 
        evaldiv0every=1, skip_avg=false)

    div0s = Array{Float64}(undef, T+2)
    for t=1:evaldiv0every:(T+1)
        div0s[t] = KL(wx0, copies_wx[:,t]) + KL(wy0, copies_wy[:,t])
    end
    div0s[T+2] = KL(wx0, avg_wx) + KL(wy0, avg_wy)
    open(logfile, "a") do f
        for t=1:evaldiv0every:(T+1)
            write(f, "KL-divergence to reference point at iteration#$t: $(div0s[t])\n")
        end
        write(f, "KL-divergence to reference point at avg iterate: $(div0s[T+2])\n")
    end

    plt_div0 = plot(range(1, stop=T, step=evaldiv0every), div0s[1:evaldiv0every:T], xlabel="k", label="")
    if !skip_avg
        hline!([div0s[T+2]], label="avg iterate")
    end
    title!("KL-divergence to reference point of iterates")
    fn = "$resultdir/div_to_refpoints.png"
    savefig(plt_div0, fn)
    
    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    plt_div0_log = plot(range(1, stop=T, step=evaldiv0every), eps .+ max.(0, div0s[1:evaldiv0every:T]), xlabel="k", yscale=:log10, label="")
    if !skip_avg
        hline!([div0s[T+2]], label="avg iterate")
    end
    title!("KL-divergence to reference point of iterates (log-linear scale)")
    fn = "$resultdir/div_to_refpoints_logscale.png"
    savefig(plt_div0_log, fn)
    
    return plt_div0, plt_div0_log, div0s
end

function plot_iters(gmat, T, copies_wx, copies_wy, avg_wx, avg_wy,
        resultdir,
        plotiterevery=5, Tstart=1)

    pltt = plot_iter_findom(gmat, avg_wx, avg_wy)
    fn = "$resultdir/iter_avg.png"
    savefig(pltt, fn)

    @showprogress 1 for t=Tstart:plotiterevery:(T+1) # minimum update interval of 1 second
    # for t=1:(T+1)
        pltt = plot_iter_findom(gmat, copies_wx[:,t], copies_wy[:,t])
        fn = "$resultdir/iteration#$t.png"
        savefig(pltt, fn)
    end
end

function plot_firstvars(gmat, wx0, wy0, gval0,
        resultdir)
    plt_firstvar_x, plt_firstvar_y = plot_first_variations_findom(gmat, wx0, wy0, gval0)
    fn = "$resultdir/firstvar_at_refpoint_x.png"
    savefig(plt_firstvar_x, fn)
    fn = "$resultdir/firstvar_at_refpoint_y.png"
    savefig(plt_firstvar_y, fn)
    plt_firstvar_x, plt_firstvar_y
end

end
