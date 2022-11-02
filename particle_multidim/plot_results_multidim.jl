module PlotResMultidim

export plot_NI_err, plot_locNI_err,
    plot_V, plot_V_xy

using ProgressMeter
using Plots
pyplot() # this sometimes takes a minute

using Random
include("../utils/multidim.jl")
using .MultidimUtils
include("CPMP_bilin_game_multidim.jl")
using .CPMPBilinGameMultidim
include("game_theory_contdom_multidim.jl")
using .GameTheoryContdomMultidimUtils
include("lyapunov_multidim.jl")
using .LyapunovMultidimUtils

"Compute and plot (global) NI error of iterates"
function plot_NI_err(gfun, T, copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y,
        resultdir, logfile, 
        evalNIevery=1;
        TTxx=10, eta0_xx=1e-3, deltaxx=5e-2, restartsxx=nothing, 
        TTyy=10, eta0_yy=1e-3, deltayy=5e-2, restartsyy=nothing)

    nierrs = Array{Float64}(undef, T+2)
    @showprogress 1 for t=1:evalNIevery:(T+1) # minimum update interval of 1 second
    # for t=1:evalNIevery:(T+1)
        nierrs[t] = glob_NI_err(gfun, copies_wx[:,t], copies_x[:,:,t], copies_wy[:,t], copies_y[:,:,t]; 
            TTxx=TTxx, eta0_xx=eta0_xx, deltaxx=deltaxx, restartsxx=restartsxx,
            TTyy=TTyy, eta0_yy=eta0_yy, deltayy=deltayy, restartsyy=restartsyy)
    end
    # nierrs[T+2] = glob_NI_err(gfun, avg_wx, avg_x, avg_wy, avg_y) # takes too long
    open(logfile, "a") do f
        for t=1:evalNIevery:(T+1)
            write(f, "glob_NI_err at iteration#$t: $(nierrs[t])\n")
        end
        # write(f, "glob_NI_err at avg iterate: $(nierrs[T+2])\n")
    end

    plt_NI = plot(range(1, stop=T+1, step=evalNIevery), nierrs[1:evalNIevery:T+1], xlabel="k", label="")
    # hline!([nierrs[T+2]], label="avg iterate")
    title!("NI error of iterates")
    fn = "$resultdir/NI_errors.png"
    savefig(plt_NI, fn)

    eps = 1e-10 # numerical stability (we only compute lower bound on the true NI error)
    plt_NI_log = plot(range(1, stop=T+1, step=evalNIevery), eps .+ max.(0, nierrs[1:evalNIevery:T+1]), xlabel="k", yscale=:log10, label="")
    # hline!([nierrs[T+2]], label="avg iterate")
    title!("NI error of iterates (log-linear scale)")
    fn = "$resultdir/NI_errors_logscale.png"
    savefig(plt_NI_log, fn)

    return plt_NI, plt_NI_log, nierrs
end

"Compute and plot local NI error of iterates"
function plot_locNI_err(gfun, T, copies_wx, copies_x, copies_wy, copies_y, avg_wx, avg_x, avg_wy, avg_y,
        resultdir, logfile, 
        evallocNIevery=1)

    locnierrs = Array{Float64}(undef, T+2)
    for t=1:evallocNIevery:(T+1)
        locnierrs[t] = loc_NI_err(gfun, copies_wx[:,t], copies_x[:,:,t], copies_wy[:,t], copies_y[:,:,t])
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
    title!("Local NI error of iterates")
    fn = "$resultdir/local_NI_errors.png"
    savefig(plt_locNI, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    plt_locNI_log = plot(range(1, stop=T+1, step=evallocNIevery), eps .+ locnierrs[1:evallocNIevery:T+1], xlabel="k", yscale=:log10, label="")
    # hline!([nierrs[T+2]], label="avg iterate")
    title!("Local NI error of iterates (log-linear scale)")
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
        alpha, lambda, tau)

    Vweis = Array{Float64}(undef, T+2)
    Vposs = Array{Float64}(undef, T+2)
    Vtots = Array{Float64}(undef, T+2)
    for t=1:evalVevery:(T+1)
        phi_mat_z = phi_mat(copies_z[:,:,t], zstar, alpha, lambda, tau)
        Vweis[t] = V_wei(copies_w[:,t], copies_z[:,:,t], wstar, zstar, eta_w;
                alpha=alpha, lambda=lambda, tau=tau,
                phi_mat_z=phi_mat_z)
        Vposs[t] = V_pos(copies_w[:,t], copies_z[:,:,t], wstar, zstar, eta_z;
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
    
    plt_V = plot(range(1, stop=T, step=evalVevery), Vtots[1:evalVevery:T], xlabel="k", label="")
    plot!(range(1, stop=T, step=evalVevery), Vweis[1:evalVevery:T], xlabel="k", line=:dash, label="weight term")
    plot!(range(1, stop=T, step=evalVevery), Vposs[1:evalVevery:T], xlabel="k", line=:dot, label="position term")
    title!("Lyapunov potential of iterates -- $z_str")
    fn = "$resultdir/V_$z_str.png"
    savefig(plt_V, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    plt_V_log = plot(range(1, stop=T, step=evalVevery), eps.+ max.(0, Vtots[1:evalVevery:T]), xlabel="k", label="", yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ max.(0, Vweis[1:evalVevery:T]), xlabel="k", line=:dash, label="weight term", yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ max.(0, Vposs[1:evalVevery:T]), xlabel="k", line=:dot, label="position term", yscale=:log10)
    # hline!([div0s[T+2]], label="avg iterate") --> not defined
    title!("Lyapunov potential of iterates -- $z_str (log-linear scale)")
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
        alpha_x, lambda_x, tau_x, alpha_y, lambda_y, tau_y)

    Vweis_xy = Array{Float64}(undef, T+2)
    Vposs_xy = Array{Float64}(undef, T+2)
    Vtots_xy = Array{Float64}(undef, T+2)
    for t=1:evalVevery:(T+1)
        phi_mat_x = phi_mat(copies_x[:,:,t], xstar, alpha_x, lambda_x, tau_x)
        phi_mat_y = phi_mat(copies_y[:,:,t], ystar, alpha_y, lambda_y, tau_y)
        Vweis_xy[t] = V_wei_xy(copies_wx[:,t], copies_x[:,:,t], copies_wy[:,t], copies_y[:,:,t], wxstar, xstar, wystar, ystar, eta_wx, eta_wy;
                alpha_x=alpha_x, lambda_x=lambda_x, tau_x=tau_x, phi_mat_x=phi_mat_x, 
                alpha_y=alpha_y, lambda_y=lambda_y, tau_y=tau_y, phi_mat_y=phi_mat_y)
        Vposs_xy[t] = V_pos_xy(copies_wx[:,t], copies_x[:,:,t], copies_wy[:,t], copies_y[:,:,t], wxstar, xstar, wystar, ystar, eta_x, eta_y;
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
    
    plt_Vxy = plot(range(1, stop=T, step=evalVevery), Vtots_xy[1:evalVevery:T], xlabel="k", label="")
    plot!(range(1, stop=T, step=evalVevery), Vweis_xy[1:evalVevery:T], xlabel="k", line=:dash, label="weight term")
    plot!(range(1, stop=T, step=evalVevery), Vposs_xy[1:evalVevery:T], xlabel="k", line=:dot, label="position term")
    title!("Lyapunov potential of iterates (for x + for y)")
    fn = "$resultdir/Vxy.png"
    savefig(plt_Vxy, fn)

    eps = 1e-30 # avoid PyPlot crashing when y-axis=0 in log plot
    plt_Vxy_log = plot(range(1, stop=T, step=evalVevery), eps.+ max.(0, Vtots_xy[1:evalVevery:T]), xlabel="k", label="", yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ max.(0, Vweis_xy[1:evalVevery:T]), xlabel="k", line=:dash, label="weight term", yscale=:log10)
    plot!(range(1, stop=T, step=evalVevery), eps.+ max.(0, Vposs_xy[1:evalVevery:T]), xlabel="k", line=:dot, label="position term", yscale=:log10)
    # hline!([div0s[T+2]], label="avg iterate") --> not defined
    title!("Lyapunov potential of iterates (for x + for y) (log-linear scale)")
    fn = "$resultdir/Vxy_logscale.png"
    savefig(plt_Vxy_log, fn)

    return plt_Vxy, plt_Vxy_log, Vweis_xy, Vposs_xy, Vtots_xy
end



end
