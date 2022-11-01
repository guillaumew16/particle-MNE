module GameTheoryFindomUtils

export NI_err, payoff, gap, KL

"""
Compute the Nikaido-Isoda (NI) error of (wx, wy) defined as
    max_{wx0, wy0} <wx|gmat|wy0> - <wx0|gmat|wy>
using that max over wy0 of <foo|wy0> is max over j of (foo)_j.
"""
function NI_err(gmat, wx, wy)
    maxx = maximum(transpose(wx) * gmat)
    miny = minimum(gmat * wy)
    return maxx - miny
end

"Payoff when (wx, wy) are played: <wx|gmat|wy>"
payoff(gmat, wx, wy) = transpose(wx) * gmat * wy

"""
Gap of the candidate (wx, wy) w.r.t reference point (wx0, wy0):
    gap(w0, w) = <wx|gmat|wy0> - <wx0|gmat|wy>
"""
gap(gmat, wx0, wy0, wx, wy) = payoff(gmat, wx, wy0) - payoff(gmat, wx0, wy)

"Kullback-Leibler divergence"
function KL(w0, w)
    @assert size(w0) == size(w)
    out = 0
    for i in eachindex(w0)
        out += w0[i] * log(w0[i]/w[i])
    end
    return out
end

end
