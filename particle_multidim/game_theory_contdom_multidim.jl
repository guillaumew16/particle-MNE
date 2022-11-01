module GameTheoryContdomMultidimUtils

export locmat, loc_NI_err, glob_NI_err, payoff, gap

using Random
using ForwardDiff

include("../utils/multidim.jl")
using .MultidimUtils

"""
Compute the local payoff matrix

x: Array dimx*thismx (not necessarily the global mx)
y: Array dimy*thismy
"""
# function locmat(gfun, x, y)
#     dimx, thismx = size(x)
#     dimy, thismy = size(y)
#     gmat = Array{Float64}(undef, thismx, thismy)
#     for i=1:thismx, j=1:thismy
#         gmat[i,j] = gfun(x[:,i], y[:,j])
#     end
#     return gmat
# end
function locmat(gfun, x, y) # somehow ForwardDiff only works using this version
    return [ gfun(x[:,i], y[:,j]) for i=1:size(x)[2], j=1:size(y)[2] ]
end

"""
For fixed positions (x, y), i.e fixed local payoff matrix locmat, compute the Nikaido-Isoda (NI) error of (wx, wy) defined as
    max_{wx0, wy0} <wx|locmat|wy0> - <wx0|locmat|wy>
using that max over wy0 of <foo|wy0> is max over j of (foo)_j.
"""
function loc_NI_err(wx, wy; locmat)
    maxx = maximum(transpose(wx) * locmat)
    miny = minimum(locmat * wy)
    return maxx - miny
end

function loc_NI_err(gfun, wx, x, wy, y)
    @assert length(wx) == size(x)[2]
    @assert length(wy) == size(y)[2]
    loc_NI_err(wx, wy; locmat=locmat(gfun, x, y))
end

"""
Compute the "global" Nikaido-Isoda (NI) error of (mu, nu) = ((wx, x), (wy, y)) defined as
    max_{mu0, nu0} <mu|F|nu0> - <mu0|F|nu>
where F is the "continuous payoff matrix",
using that max over nu0 of <foo|nu0> is max over y of foo(y).

The max is estimated by running `TTyy` iters of GD with stepsize `eta0_yy`,
    - `restartsyy` times with random uniform initialization, OR
    - `dimy^(1/deltayy)` times with uniform grid initialization
Among `deltayy` and `restartsyy`, exactly one of the two must be specified, the other must be set to `nothing`

NB: what this function computes is a lower-bound on the global NI error...
"""
function glob_NI_err(gfun, wx, x, wy, y;
        TTxx=10, eta0_xx=1e-3, deltaxx=5e-2, restartsxx=nothing, 
        TTyy=10, eta0_yy=1e-3, deltayy=5e-2, restartsyy=nothing, 
        rng=MersenneTwister(1234)
)
    dimx, mx = size(x)
    dimy, my = size(y)

    @assert xor( isnothing(deltaxx), isnothing(restartsxx) )
    @assert xor( isnothing(deltayy), isnothing(restartsyy) )

    # estimate max_{nu0} <mu|F|nu0>
    negmuF(yj) = -sum( wx[i] * gfun(x[:,i], yj) for i in eachindex(wx) )
    if !isnothing(restartsyy)
        yy0s = rand(rng, dimy, restartsyy)
    else
        yy0s = multidim_grid(dimy, deltayy)
    end
    yy0s = hcat(y, yy0s) # also try starting exactly at the current iterates (reasonable guess if iterates are close to optimum)
    min_negmuF, yyopt = restarted_GD(negmuF, dimy, TTyy, eta0_yy, yy0s)
    max_muF = -min_negmuF

    # estimate min_{mu0} <mu0|F|nu>
    Fnu(xi) = sum( wy[j] * gfun(xi, y[:,j]) for j in eachindex(wy) )
    if !isnothing(restartsxx)
        xx0s = rand(rng, dimx, restartsxx)
    else
        xx0s = multidim_grid(dimx, deltaxx)
    end
    xx0s = hcat(x, xx0s)
    min_Fnu, xxopt = restarted_GD(Fnu, dimx, TTxx, eta0_xx, xx0s)

    out = max_muF - min_Fnu
    return out = max(0, out) # in case the lower bound we computed turns out to be even worse than just 0...
end

function restarted_GD(f, dim, TT, eta0_zz, zz0s)
    min_f = Inf
    zzopt = Array{Float64}(undef, dim)
    for i in size(zz0s)[2]
        # run gradient descent for TT steps starting from zz0s[:,i]
        zz = zz0s[:,i]
        eta_zz = eta0_zz # cst stepsize
        for tt=1:TT
            gdt = ForwardDiff.gradient(f, zz)
            zz = zz - eta_zz * gdt
            if f(zz) < min_f
                zzopt, min_f = zz, f(zz)
            end
        end
    end
    return min_f, zzopt
end

"Payoff when (wx, x, wy, y) are played: <mu|F|nu>"
payoff(gfun, wx, x, wy, y) = transpose(wx) * locmat(gfun, x, y) * wy

"""
Gap of the candidate (wx, wy) w.r.t reference point (wx0, wy0):
    gap(mu0, nu0, mu, nu) = <mu|F|nu0> - <mu0|F|nu>
"""
gap(gfun, wx0, x0, wy0, y0, wx, x, wy, y) = payoff(gfun, wx, x, wy0, y0) - payoff(gfun, wx0, x0, wy, y)

end
