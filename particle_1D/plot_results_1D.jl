module PlotRes1D

export plot_NI_err, plot_locNI_err,
    plot_V, plot_V_xy,
    plot_iters, plot_firstvars

using ProgressMeter
using Plots
pyplot() # this sometimes takes a minute

using Random
include("../utils/misc.jl")
using .MiscUtils
include("CPMP_bilin_game_1D.jl")
using .CPMPBilinGame1D
include("game_theory_contdom_1D.jl")
using .GameTheoryContdom1DUtils
include("lyapunov_1D.jl")
using .Lyapunov1DUtils
include("plot_game_1D.jl")
using .PlotGame1DUtils

"Compute and plot (global) NI error of iterates"
function plot_NI_err(gfun, T, copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y,
        resultdir, logfile, 
        evalNIevery=1;
        deltax=5e-3, deltay=5e-3,
        skip_avg=false, hidetitle=false,
        xmin=0., xmax=1., ymin=0., ymax=1.)

    nierrs = Array{Float64}(undef, T+2)
    @showprogress 1 for t=1:evalNIevery:(T+1) # minimum update interval of 1 second
    # for t=1:evalNIevery:(T+1)
        nierrs[t] = glob_NI_err(gfun, copies_wx[:,t], copies_x[:,t], copies_wy[:,t], copies_y[:,t]; deltax=deltax, deltay=deltay, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    end
    if !skip_avg
        nierrs[T+2] = glob_NI_err(gfun, avg_wx, avg_x, avg_wy, avg_y; deltax=deltax, deltay=deltay, extrahacky=true, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)  # takes a bit too long to compute if extrahacky=false: T*(mx/deltay+my/deltax)
    end
    open(logfile, "a") do f
        for t=1:evalNIevery:(T+1)
            write(f, "glob_NI_err at iteration#$t: $(nierrs[t])\n")
        end
        if !skip_avg 
            write(f, "glob_NI_err at avg iterate: $(nierrs[T+2])\n") 
        end
    end

    plt_NI = plot(range(1, stop=T+1, step=evalNIevery), nierrs[1:evalNIevery:T+1], xlabel="k", label="")
    if !skip_avg
        hline!([nierrs[T+2]], label="avg iterate")
    end
    if !hidetitle
        title!("NI error of iterates")
    end
    fn = "$resultdir/NI_errors.png"
    savefig(plt_NI, fn)

    eps = 1e-10 # numerical stability (we use approximations (with deltax, deltay) to compute glob_NI_err)
    plt_NI_log = plot(range(1, stop=T+1, step=evalNIevery), eps .+ max.(0, nierrs[1:evalNIevery:T+1]), xlabel="k", yscale=:log10, label="")
    if !skip_avg
        hline!([ eps + max(0, nierrs[T+2]) ], label="avg iterate")
    end
    if !hidetitle
        title!("NI error of iterates (log-linear scale)")
    end
    fn = "$resultdir/NI_errors_logscale.png"
    savefig(plt_NI_log, fn)

    return plt_NI, plt_NI_log, nierrs
end

"Compute and plot local NI error of iterates"
function plot_locNI_err(gfun, T, copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y,
        resultdir, logfile,
        evallocNIevery=1,
        hidetitle=false)

    locnierrs = Array{Float64}(undef, T+2)
    for t=1:evallocNIevery:(T+1)
        locnierrs[t] = loc_NI_err(gfun, copies_wx[:,t], copies_x[:,t], copies_wy[:,t], copies_y[:,t])
    end
    # locnierrs[T+2] = loc_NI_err(avg_wx, avg_x, avg_wy, avg_y) # takes way too long to compute: mx*my*T^2
    open(logfile, "a") do f
        for t=1:evallocNIevery:(T+1)
            write(f, "loc_NI_err at iteration#$t: $(locnierrs[t])\n")
        end
        # write(f, "loc_NI_err at avg iterate: $(locnierrs[T+2])\n")
    end

    plt_locNI = plot(range(1, stop=T+1, step=evallocNIevery), locnierrs[1:evallocNIevery:T+1], xlabel="k", label="")
    # hline!([nierrs[T+2]], label="avg iterate")
    if !hidetitle
        title!("Local NI error of iterates")
    end
    fn = "$resultdir/local_NI_errors.png"
    savefig(plt_locNI, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    plt_locNI_log = plot(range(1, stop=T+1, step=evallocNIevery), eps .+ max.(0, locnierrs[1:evallocNIevery:T+1]), xlabel="k", yscale=:log10, label="")
    # hline!([nierrs[T+2]], label="avg iterate")
    if !hidetitle
        title!("Local NI error of iterates (log-linear scale)")
    end
    fn = "$resultdir/local_NI_errors_logscale.png"
    savefig(plt_locNI_log, fn)

    return plt_locNI, plt_locNI_log, locnierrs
end

"Compute and plot weight and position and total lyapunov potential of iterates (wx,x variables OR wy,y variables)"
function plot_V(gfun, T, copies_w, copies_z, avg_w, avg_z,
        wstar, zstar,
        eta_w, eta_z, # assume we used constant step-sizes
        resultdir, logfile, z_str="x", 
        evalVevery=1;
        hidetitle=false,
        alpha, lambda, tau)

    Vweis = Array{Float64}(undef, T+2)
    Vposs = Array{Float64}(undef, T+2)
    Vtots = Array{Float64}(undef, T+2)
    for t=1:evalVevery:(T+1)
        phi_mat_z = phi_mat(copies_z[:,t], zstar, alpha, lambda, tau)
        Vweis[t] = V_wei(copies_w[:,t], copies_z[:,t], wstar, zstar, eta_w;
                alpha=alpha, lambda=lambda, tau=tau,
                phi_mat_z=phi_mat_z)
        Vposs[t] = V_pos(copies_w[:,t], copies_z[:,t], wstar, zstar, eta_z;
                alpha=alpha, lambda=lambda, tau=tau, 
                phi_mat_z=phi_mat_z)
        Vtots[t] = Vweis[t] + Vposs[t]
    end
    # Vweis[t] = V_wei(avg_w, avg_z; wstar, zstar, alpha, lambda, tau, eta_w) --> nah whatever
    open(logfile, "a") do f
        for t=1:evalVevery:(T+1)
            write(f, "V_wei_$z_str, V_pos_$z_str, V_tot_$z_str at iteration#$t: $(Vweis[t]), $(Vposs[t]), $(Vtots[t])\n")
        end
    end
    
    # plt_V = plot(range(1, stop=T, step=evalVevery), Vtots[1:evalVevery:T], xlabel="k", label="total")
    # plot!(range(1, stop=T, step=evalVevery), Vweis[1:evalVevery:T], xlabel="k", line=:dash, label="weight term")
    # plot!(range(1, stop=T, step=evalVevery), Vposs[1:evalVevery:T], xlabel="k", line=:dot, label="position term")
    plt_V = plot(range(1, stop=T, step=evalVevery), Vtots[1:evalVevery:T], xlabel="k", label="total", linewidth=3, linealpha=0.75)
    plot!(range(1, stop=T, step=evalVevery), Vweis[1:evalVevery:T], xlabel="k", line=:dot, label="weight term", linewidth=2)
    plot!(range(1, stop=T, step=evalVevery), Vposs[1:evalVevery:T], xlabel="k", line=:dash, label="position term", linewidth=1)
    if !hidetitle
        title!("Lyapunov potential of iterates -- $z_str")
    end
    fn = "$resultdir/V_$z_str.png"
    savefig(plt_V, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    # plt_V_log = plot(range(1, stop=T, step=evalVevery), eps.+ Vtots[1:evalVevery:T], xlabel="k", label="total", yscale=:log10)
    # plot!(range(1, stop=T, step=evalVevery), eps.+ Vweis[1:evalVevery:T], xlabel="k", line=:dash, label="weight term", yscale=:log10)
    # plot!(range(1, stop=T, step=evalVevery), eps.+ Vposs[1:evalVevery:T], xlabel="k", line=:dot, label="position term", yscale=:log10)
    plt_V_log = plot(range(1, stop=T, step=evalVevery), eps.+ Vtots[1:evalVevery:T], xlabel="k", label="total", linewidth=3, linealpha=0.75, yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ Vweis[1:evalVevery:T], xlabel="k", line=:dot, label="weight term", linewidth=2, yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ Vposs[1:evalVevery:T], xlabel="k", line=:dash, label="position term", linewidth=1, yscale=:log10)
    # hline!([div0s[T+2]], label="avg iterate") --> not defined
    if !hidetitle
        title!("Lyapunov potential of iterates -- $z_str (log-linear scale)")
    end
    fn = "$resultdir/V_$(z_str)_logscale.png"
    savefig(plt_V_log, fn)

    return plt_V, plt_V_log, Vweis, Vposs, Vtots
end

"Compute and plot weight and position and total lyapunov potential of iterates (for x + for y)"
function plot_V_xy(gfun, T, copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y,
        wxstar, xstar, wystar, ystar,
        eta_wx, eta_x, eta_wy, eta_y, # assume we used constant step-sizes
        resultdir, logfile, 
        evalVevery=1;
        hidetitle=false,
        alpha_x, lambda_x, tau_x, alpha_y, lambda_y, tau_y)

    Vweis_xy = Array{Float64}(undef, T+2)
    Vposs_xy = Array{Float64}(undef, T+2)
    Vtots_xy = Array{Float64}(undef, T+2)
    for t=1:evalVevery:(T+1)
        phi_mat_x = phi_mat(copies_x[:,t], xstar, alpha_x, lambda_x, tau_x)
        phi_mat_y = phi_mat(copies_y[:,t], ystar, alpha_y, lambda_y, tau_y)
        Vweis_xy[t] = V_wei_xy(copies_wx[:,t], copies_x[:,t], copies_wy[:,t], copies_y[:,t], wxstar, xstar, wystar, ystar, eta_wx, eta_wy;
                alpha_x=alpha_x, lambda_x=lambda_x, tau_x=tau_x, phi_mat_x=phi_mat_x, 
                alpha_y=alpha_y, lambda_y=lambda_y, tau_y=tau_y, phi_mat_y=phi_mat_y)
        Vposs_xy[t] = V_pos_xy(copies_wx[:,t], copies_x[:,t], copies_wy[:,t], copies_y[:,t], wxstar, xstar, wystar, ystar, eta_x, eta_y;
                alpha_x=alpha_x, lambda_x=lambda_x, tau_x=tau_x, phi_mat_x=phi_mat_x,
                alpha_y=alpha_y, lambda_y=lambda_y, tau_y=tau_y, phi_mat_y=phi_mat_y)
        Vtots_xy[t] = Vweis_xy[t] + Vposs_xy[t]
    end
    # Vweis[t] = V_wei_xy(...) --> nah whatever
    open(logfile, "a") do f
        for t=1:evalVevery:(T+1)
            write(f, "V_wei_xy, V_pos_xy, V_tot_xy at iteration#$t: $(Vweis_xy[t]), $(Vposs_xy[t]), $(Vtots_xy[t])\n")
        end
    end
    
    # plt_Vxy = plot(range(1, stop=T, step=evalVevery), Vtots_xy[1:evalVevery:T], xlabel="k", label="total")
    # plot!(range(1, stop=T, step=evalVevery), Vweis_xy[1:evalVevery:T], xlabel="k", line=:dash, label="weight term")
    # plot!(range(1, stop=T, step=evalVevery), Vposs_xy[1:evalVevery:T], xlabel="k", line=:dot, label="position term")
    plt_Vxy = plot(range(1, stop=T, step=evalVevery), Vtots_xy[1:evalVevery:T], xlabel="k", label="total", linewidth=3, linealpha=0.75)
    plot!(range(1, stop=T, step=evalVevery), Vweis_xy[1:evalVevery:T], xlabel="k", line=:dot, label="weight term", linewidth=2)
    plot!(range(1, stop=T, step=evalVevery), Vposs_xy[1:evalVevery:T], xlabel="k", line=:dash, label="position term", linewidth=1)
    if !hidetitle
        title!("Lyapunov potential of iterates (for x + for y)")
    end
    fn = "$resultdir/Vxy.png"
    savefig(plt_Vxy, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    # plt_Vxy_log = plot(range(1, stop=T, step=evalVevery), eps.+ Vtots_xy[1:evalVevery:T], xlabel="k", label="total", yscale=:log10)
    # plot!(range(1, stop=T, step=evalVevery), eps.+ Vweis_xy[1:evalVevery:T], xlabel="k", line=:dash, label="weight term", yscale=:log10)
    # plot!(range(1, stop=T, step=evalVevery), eps.+ Vposs_xy[1:evalVevery:T], xlabel="k", line=:dot, label="position term", yscale=:log10)
    plt_Vxy_log = plot(range(1, stop=T, step=evalVevery), eps.+ Vtots_xy[1:evalVevery:T], xlabel="k", label="total", linewidth=3, linealpha=0.75, yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ Vweis_xy[1:evalVevery:T], xlabel="k", line=:dot, label="weight term", linewidth=2, yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ Vposs_xy[1:evalVevery:T], xlabel="k", line=:dash, label="position term", linewidth=1, yscale=:log10)
    # hline!([div0s[T+2]], label="avg iterate") --> not defined
    if !hidetitle
        title!("Lyapunov potential of iterates (for x + for y) (log-linear scale)")
    end
    fn = "$resultdir/Vxy_logscale.png"
    savefig(plt_Vxy_log, fn)

    return plt_Vxy, plt_Vxy_log, Vweis_xy, Vposs_xy, Vtots_xy
end

function plot_iters(gfun, T, copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y,
        resultdir,
        plotiterevery=5, Tstart=1;
        len_xs=401, len_ys=401,
        xmin=0., xmax=1., ymin=0., ymax=1., clip=true)
    xs = range(xmin, stop=xmax, length=len_xs)
    ys = range(ymin, stop=ymax, length=len_ys)
    gfuns = locmat(gfun, xs, ys)

    copies_x01 = linrescale(copies_x, xmin, xmax, clip=clip)
    copies_y01 = linrescale(copies_y, ymin, ymax, clip=clip)
    avg_x01 = linrescale(copies_x, xmin, xmax, clip=clip)
    avg_y01 = linrescale(copies_y, ymin, ymax, clip=clip)
    copies_smoothsig_wx, copies_smoothsig_wy, avg_smoothsig_wx, avg_smoothsig_wy = smoothen_iters(copies_wx, copies_x01, copies_wy, copies_y01,
        avg_wx, avg_x01, avg_wy, avg_y01;
        len_xs=len_xs, len_ys=len_ys)
    pltt = plot_iter_smooth_1D(avg_wx, avg_x, avg_wy, avg_y, 
        avg_smoothsig_wx, avg_smoothsig_wy;
        gfuns=gfuns,
        showlines=false,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    fn = "$resultdir/iter_avg.png"
    savefig(pltt, fn)

    @showprogress 1 for t=Tstart:plotiterevery:(T+1) # minimum update interval of 1 second
    # for t=1:(T+1)
        pltt = plot_iter_smooth_1D(copies_wx[:,t], copies_x[:,t], copies_wy[:,t], copies_y[:,t],
            copies_smoothsig_wx[:,t], copies_smoothsig_wy[:,t];
            gfuns=gfuns,
            showlines=true,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        fn = "$resultdir/iteration#$t.png"
        savefig(pltt, fn)
    end

    return copies_smoothsig_wx, copies_smoothsig_wy, avg_smoothsig_wx, avg_smoothsig_wy
end

function plot_firstvars(gfun, wx0, x0, wy0, y0, gval0,
        resultdir;
        hidetitle=false, hidelabels=false, hidestems=false,
        len_xs=401, len_ys=401,
        xmin=0., xmax=1., ymin=0., ymax=1.)

    plt_firstvar_x, plt_firstvar_y = plot_first_variations_1D(gfun, wx0, x0, wy0, y0, gval0;
        hidetitle=hidetitle, hidelabels=hidelabels, hidestems=hidestems,
        len_xs=len_xs, len_ys=len_ys, 
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    fn = "$resultdir/firstvar_x.png"
    savefig(plt_firstvar_x, fn)
    fn = "$resultdir/firstvar_y.png"
    savefig(plt_firstvar_y, fn)
    plt_firstvar_x, plt_firstvar_y
end


end
