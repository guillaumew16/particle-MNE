module LyapunovMultidimUtils

export KL, divgce_pos_naive, divgce_naive, 
    psi, phi_mat, 
    agg_wei, agg_pos_mean, agg_pos_tracecov, 
    V_wei, V_pos, V_tot, wei0,
    V_wei_xy, V_pos_xy, V_tot_xy, wei0_xy

include("../utils/multidim.jl")
using .MultidimUtils: sq_torus_dist
using LogExpFunctions: xlogx, xlogy

"Kullback-Leibler divergence"
function KL(w0, w)
    @assert size(w0) == size(w)
    out = 0
    for i in eachindex(w0)
        # out += w0[i] * log(w0[i]/w[i])
        out += xlogx(w0[i]) - xlogy(w0[i], w[i])
    end
    return out
end

function divgce_pos_naive(w0, z0, w, z; eta_w, eta_z)
    @assert size(z0) == size(z)
    dim, m = size(z)
    s = 0
    for i=1:m
        disti_sq = sq_torus_dist(z0[:,i], z[:,i])
        s += w[i] * disti_sq / 2
    end
    return s*eta_w/eta_z
end

function divgce_naive(w0, z0, w, z; eta_w, eta_z)
    return KL(w0, w) + divgce_pos_naive(w0, z0, w, z; eta_w=eta_w, eta_z=eta_z)
end

function psi(r; alpha, lambda)
    if r > lambda
        return 0
    elseif r < 0
        error("r<0")
    else
        return exp(-r^alpha / alpha)
    end
end

"""
Compute the matrix phi_{Ii} = phi_I(z_i) = psi(||zstar_I - z_i||/tau; lambda)
"""
function phi_mat(z, zstar, alpha, lambda, tau; tol=0.)
    dim, mstar = size(zstar)
    dim, m = size(z)
    out = zeros(mstar, m)
    for I=1:mstar
        for i=1:m
            disti = sqrt( sq_torus_dist(zstar[:,I], z[:,i]) )
            out[I,i] = psi(disti/tau; alpha, lambda)
        end
    end

    # check that the phi_I have disjoint supports, or approximately so
    if tol == 0
        dzstar_min = sqrt( minimum([sq_torus_dist(zstar[:,I], zstar[:,J]) for I=1:mstar for J=1:I-1]) )
        if lambda*tau > dzstar_min/2
            error("phi_I do not have disjoint supports")
        end
    elseif any(s -> s>1+tol, sum(out, dims=1))
        error("phi_I do not have disjoint supports")
    end

    return out
end

function agg_wei(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=nothing)
    if isnothing(phi_mat_z)
        phi_mat_z = phi_mat(z, zstar, alpha, lambda, tau)
    end
    return phi_mat_z * w
end

function agg_pos_mean(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=nothing)
    if isnothing(phi_mat_z)
        phi_mat_z = phi_mat(z, zstar, alpha, lambda, tau)
    end
    w_agg = agg_wei(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=phi_mat_z)

    dim, mstar = size(zstar)
    dim, m = size(z)
    out = zeros(dim, mstar)
    for I=1:mstar
        for i=1:m
            out[:,I] += phi_mat_z[I,i] * w[i] * z[:,i]
        end
        out[:,I] /= w_agg[I]
    end
    return out
end

"Trace(Sigma_I) for each I"
function agg_pos_tracecov(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=nothing)
    if isnothing(phi_mat_z)
        phi_mat_z = phi_mat(z, zstar, alpha, lambda, tau)
    end
    w_agg = agg_wei(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=phi_mat_z)
    z_agg = agg_pos_mean(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=phi_mat_z)
    
    mstar = length(zstar)
    m = length(z)
    out = zeros(mstar)
    for I=1:mstar
        for i=1:m
            out[I] += phi_mat_z[I,i] * w[i] * sq_torus_dist(z[:,i], z_agg[:,I])
        end
        out[I] /= w_agg[I]
    end
    return out
end

function V_wei(w, z, wstar, zstar, eta_w;
        alpha, lambda, tau,
        phi_mat_z=nothing)
    w_agg = agg_wei(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=phi_mat_z)
    return KL(wstar, w_agg) / eta_w
end

function wei0(w, z, wstar, zstar, eta_w;
        alpha, lambda, tau,
        phi_mat_z=nothing)
    w_agg = agg_wei(w, z, wstar, zstar; alpha, lambda, tau, phi_mat_z=phi_mat_z)
    return (1-sum(w_agg)) / eta_w
end

function V_pos(w, z, wstar, zstar, eta_z;
        alpha, lambda, tau,
        phi_mat_z=nothing)
    if isnothing(phi_mat_z)
        phi_mat_z = phi_mat(z, zstar, alpha, lambda, tau)
    end
    s = 0
    dim, mstar = size(zstar)
    dim, m = size(z)
    for I=1:mstar
        for i=1:m
            disti_sq = sq_torus_dist(zstar[:,I], z[:,i])
            s += w[i] * phi_mat_z[I,i] * disti_sq / 2
        end
    end
    return s / eta_z
end

function V_tot(w, z, wstar, zstar, eta_w, eta_z;
        alpha, lambda, tau,
        phi_mat_z=nothing)
    if isnothing(phi_mat_z)
        phi_mat_z = phi_mat(z, zstar, alpha, lambda, tau)
    end
    Vwei_wz = V_wei(w, z, wstar, zstar, eta_w;
        alpha=alpha, lambda=lambda, tau=tau, phi_mat_z=phi_mat_z)
    Vpos_wz = V_pos(w, z, wstar, zstar, eta_z;
        alpha=alpha, lambda=lambda, tau=tau, phi_mat_z=phi_mat_z)
    return Vwei_wz + Vpos_wz
end

function V_wei_xy(wx, x, wy, y, wxstar, xstar, wystar, ystar, eta_wx, eta_wy; 
    alpha_x, lambda_x, tau_x, phi_mat_x=nothing, 
    alpha_y, lambda_y, tau_y, phi_mat_y=nothing)
return V_wei(wx, x, wxstar, xstar, eta_wx; alpha=alpha_x, lambda=lambda_x, tau=tau_x, phi_mat_z=phi_mat_x) + V_wei(wy, y, wystar, ystar, eta_wy; alpha=alpha_y, lambda=lambda_y, tau=tau_y, phi_mat_z=phi_mat_y)
end
function wei0_xy(wx, x, wy, y, wxstar, xstar, wystar, ystar, eta_wx, eta_wy; 
    alpha_x, lambda_x, tau_x, phi_mat_x=nothing, 
    alpha_y, lambda_y, tau_y, phi_mat_y=nothing)
return wei0(wx, x, wxstar, xstar, eta_wx; alpha=alpha_x, lambda=lambda_x, tau=tau_x, phi_mat_z=phi_mat_x) + wei0(wy, y, wystar, ystar, eta_wy; alpha=alpha_y, lambda=lambda_y, tau=tau_y, phi_mat_z=phi_mat_y)
end
function V_pos_xy(wx, x, wy, y, wxstar, xstar, wystar, ystar, eta_x, eta_y; 
    alpha_x, lambda_x, tau_x, phi_mat_x=nothing, 
    alpha_y, lambda_y, tau_y, phi_mat_y=nothing)
return V_pos(wx, x, wxstar, xstar, eta_x; alpha=alpha_x, lambda=lambda_x, tau=tau_x, phi_mat_z=phi_mat_x) + V_pos(wy, y, wystar, ystar, eta_y; alpha=alpha_y, lambda=lambda_y, tau=tau_y, phi_mat_z=phi_mat_y)
end
function V_tot_xy(wx, x, wy, y, wxstar, xstar, wystar, ystar, eta_wx, eta_x, eta_wy, eta_y; 
    alpha_x, lambda_x, tau_x, phi_mat_x=nothing, 
    alpha_y, lambda_y, tau_y, phi_mat_y=nothing)
return V_tot(wx, x, wxstar, xstar, eta_wx, eta_x; alpha=alpha_x, lambda=lambda_x, tau=tau_x, phi_mat_z=phi_mat_x) + V_tot(wy, y, wystar, ystar, eta_wy, eta_y; alpha=alpha_y, lambda=lambda_y, tau=tau_y, phi_mat_z=phi_mat_y)
end

end
