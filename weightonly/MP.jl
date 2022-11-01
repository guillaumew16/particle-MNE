module MP

export run_MP, avg_iter

using ForwardDiff
include("../utils/misc.jl")
using .MiscUtils:normalize!

"""
Take a step of particle mirror descent-ascent on min_{wx} max_{wy} obj(wx, wy)
- starting from wx,wy
- evaluating the gradients at wxp,wyp
- with stepsizes eta_wx,eta_wy
Returns the updated weights wx1,wy1
"""
function step_MDA(
        obj,
        wx, wy,
        wxp, wyp,
        eta_wx, eta_wy
)
    wx1 = similar(wx)
    wy1 = similar(wy)

    # take step: wx
    s = ForwardDiff.gradient(wx -> obj(wx, wyp), wxp)
    wx1 = wx .* exp.(-eta_wx*s)
    normalize!(wx1)

    # take step: wy
    s = ForwardDiff.gradient(wy -> obj(wxp, wy), wyp)
    wy1 = wy .* exp.(eta_wy*s)
    normalize!(wy1)

    wx1, wy1
end

"""
Run (min-max) mirror-prox (MP) for T iterations on min_{wx} max_{wy} obj(wx, wy)
    eta0_wx, eta0_wy: initial step-size
        actually we use constant step-size (for now)
    extrasteps (default=2): number of steps to take to approximately solve inner variational problem
        extrasteps=1: MDA, extrasteps=2: MP

Returns
    copies_wx: Array mx*(T+1) containing the iterates (t=1 is at initialization)
    copies_wy: Array my*(T+1)
"""
function run_MP(obj, T, mx, my, eta0_wx, eta0_wy; extrasteps=2, wx00=nothing, wy00=nothing)
    ## init
    wx = ones(mx) ; normalize!(wx)
    wy = ones(my) ; normalize!(wy)
    wx = isnothing(wx00) ? wx : wx00
    wy = isnothing(wy00) ? wy : wy00
    @assert mx==length(wx) && my==length(wy)
    
    # for extragradient step
    wxp = similar(wx)
    wyp = similar(wy)

    # to store intermediate values
    copies_wx = Array{Float64}(undef, mx, T+1)
    copies_wy = Array{Float64}(undef, my, T+1)

    eta_wx, eta_wy = eta0_wx, eta0_wy # constant stepsizes

    ## loop
    # @showprogress 1 for t=1:T # minimum update interval of 1 second
    for t=1:T
        copies_wx[:,t] = copy(wx)
        copies_wy[:,t] = copy(wy)

        # extragradient ("ghost" step)
        wxp, wyp = wx, wy
        for s=1:extrasteps
            wxp, wyp = step_MDA(
                obj,
                wx, wy,
                wxp, wyp,
                eta_wx, eta_wy
            )
        end

        # take step
        wx, wy = wxp, wyp
    end

    copies_wx[:,T+1] = copy(wx)
    copies_wy[:,T+1] = copy(wy)

    return copies_wx, copies_wy
end

function avg_iter(copies_wx, copies_wy)
    T = size(copies_wx)[2] - 1
    avg_wx = sum(copies_wx, dims=2) / (T+1)
    avg_wy = sum(copies_wy, dims=2) / (T+1)
    avg_wx, avg_wy = avg_wx[:], avg_wy[:] # convert (*×1) Matrix to Vector
    return avg_wx, avg_wy
end

end
