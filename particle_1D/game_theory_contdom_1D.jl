module GameTheoryContdom1DUtils

export locmat, locmat_hacky, loc_NI_err, glob_NI_err, payoff, gap

include("../utils/misc.jl")
using .MiscUtils: linrescale

"""
Compute the local payoff matrix
"""
function locmat(gfun, x, y)
    gmat = Array{Real}(undef, length(x), length(y))
    for i in eachindex(x), j in eachindex(y)
        gmat[i,j] = gfun(x[i], y[j])
    end
    return gmat
end

# dict_apcontmats = Dict((deltax, deltay) => apcontmat)
dict_apcontmats = Dict()
"""
Return an approximation of the local payoff matrix based on a precomputed `apcontmat` (approximate continuous payoff matrix)
"""
function locmat_hacky(gfun, x, y; deltax=deltax, deltay=deltay,
        xmin::Real=0., xmax::Real=1., ymin::Real=0., ymax::Real=1.)
    @assert all(>=(xmin), x) && all(<=(xmax), x) 
    @assert all(>=(ymin), y) && all(<=(ymax), y)
    apcontx = range(xmin, stop=xmax, step=deltax)
    apconty = range(ymin, stop=ymax, step=deltay)
    lenx, leny = length(apcontx), length(apconty)
    if haskey(dict_apcontmats, (deltax, deltay))
        apcontmat = dict_apcontmats[(deltax, deltay)]
    else
        apcontmat = locmat(gfun, apcontx, apconty)
        dict_apcontmats[(deltax, deltay)] = apcontmat
    end
    gmat = Array{Float64}(undef, length(x), length(y))
    x01 = linrescale(x, xmin, xmax)
    y01 = linrescale(y, ymin, ymax)
    pos2idx(pos, len) = ( @assert 0<=pos && pos<1 ; Int(floor(len*pos)) % len + 1)
    for i in eachindex(x), j in eachindex(y)
        gmat[i,j] = apcontmat[pos2idx(x01[i], lenx), pos2idx(y01[j], leny)]
    end
    return gmat
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
    @assert length(wx) == length(x)
    @assert length(wy) == length(y)
    loc_NI_err(wx, wy; locmat=locmat(gfun, x, y))
end

"""
Compute the "global" Nikaido-Isoda (NI) error of (mu, nu) = ((wx, x), (wy, y)) defined as
    max_{mu0, nu0} <mu|F|nu0> - <mu0|F|nu>
where F is the "continuous payoff matrix",
using that max over nu0 of <foo|nu0> is max over y of foo(y).
"""
function glob_NI_err(gfun, wx, x, wy, y; deltax=5e-3, deltay=5e-3, extrahacky=false,
        xmin::Real=0., xmax::Real=1., ymin::Real=0., ymax::Real=1.)

    @assert all(>=(xmin), x) && all(<=(xmax), x) 
    @assert all(>=(ymin), y) && all(<=(ymax), y)
    apcontx = range(xmin, stop=xmax, step=deltax)
    apconty = range(ymin, stop=ymax, step=deltay)
    if !extrahacky
        apcont_muF = transpose(wx) * locmat(gfun, x, apconty) # approximate continuous first-variation functions (horrible hack)
        apcont_Fnu = locmat(gfun, apcontx, y) * wy
    else
        apcont_muF = transpose(wx) * locmat_hacky(gfun, x, apconty; deltax=deltax, deltay=deltay) # use an approximation to compute the approximate continuous first-variation (even worse hack)
        apcont_Fnu = locmat_hacky(gfun, apcontx, y; deltax=deltax, deltay=deltay) * wy
    end
    maxx = maximum(apcont_muF)
    miny = minimum(apcont_Fnu)
    return maxx - miny
end

"Payoff when (wx, x, wy, y) are played: <mu|F|nu>"
payoff(gfun, wx, x, wy, y) = transpose(wx) * locmat(gfun, x, y) * wy

"""
Gap of the candidate (wx, wy) w.r.t reference point (wx0, wy0):
    gap(mu0, nu0, mu, nu) = <mu|F|nu0> - <mu0|F|nu>
"""
gap(gfun, wx0, x0, wy0, y0, wx, x, wy, y) = payoff(gfun, wx, x, wy0, y0) - payoff(gfun, wx0, x0, wy, y)

end
