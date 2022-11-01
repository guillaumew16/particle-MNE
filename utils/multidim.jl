module MultidimUtils

export multidim_grid, pos2idx, initialize_positions_grid_unif, sq_torus_dist, aggregate_particles_multidim

using LinearAlgebra: dot

"Grid of [0,1]^dim with step delta in each dimension. Array dim*(1/delta^dim)"
function multidim_grid(dim, delta)
    grid_1d = range(0, stop=1, step=delta)[1:end-1] # grid [0, 1)
    grid = Iterators.product((grid_1d for _=1:dim)...)
    out = [Tuple(point) for point in grid]
    out = reinterpret(Float64, out)
    return reshape(out, dim, :)
end

function pos2idx(pos; delta)
    @assert ndims(pos) == 1
    dim = length(pos)
    len = length(range(0, 1, step=delta)[1:end-1])
    pos2idx_1d(pos_1d) = ( @assert 0<=pos_1d && pos_1d<1 ; Int(floor(len*pos_1d)) % len + 1 )
    s = 0
    for k=1:dim
        s += len^(k-1) * pos2idx_1d(pos[k])
    end
    return s
end

"Higher integer d-th root: the smallest integer m such that m^d >= n. (Compare the built-in Base.isqrt.)"
iroot_higher(d, n) = Int(ceil(n^(1/d)))

function initialize_positions_grid_unif(dimx, mx)
    if dimx == 1
        mxx = mx
        x = collect(range(0, stop=1, length=mx+1)[1:mx])
        x = reshape(x, 1, size(x)...)
    else
        mxx = iroot_higher(dimx, mx)^dimx
        x = collect(multidim_grid(dimx, mxx^(-1/dimx)))
    end
    x, mxx
end

"Squared Euclidean distance between two points zi0, zi for the torus geometry on [0,1)^dim"
function sq_torus_dist(zi0, zi)
    dim, = size(zi)
    disti_sq = 0
    for k=1:dim
        distik = zi0[k] - zi[k]
        distik = min(abs(distik), abs(distik+1), abs(distik-1))
        disti_sq += distik^2
    end
    disti_sq
end

"""
Given a reference point wx0,x0 encoded with mx particles, compute wxstar,xstar and mxstar such that nearby particles are aggregated into one.
Of course this only makes sense to do if wx0,x0 correponds to a sparse measure (sparser than mx).
    thresh_pos: two particles i, i' (in [mx]) are aggregated together if dist(xi, xi') < thresh

Returns wxstar, xstar, mxstar
"""
function aggregate_particles_multidim(wx0, x0; thresh_pos=1e-5)
    dim, mx = size(x0)
    xstar = similar(x0)
    mxstar = 0
    for i=1:mx
        duplicate = false
        for ii=1:mxstar
            if sq_torus_dist(xstar[:,ii], x0[:,i]) < thresh_pos^2
                duplicate = true
                break
            end
        end
        if !duplicate
            xstar[:, mxstar+1] = x0[:, i]
            mxstar += 1
        end
    end
    xstar = xstar[:, 1:mxstar]
        
    wxstar = zeros(mxstar)
    for i=1:mx, ii=1:mxstar
        if sq_torus_dist(xstar[:,ii], x0[:,i]) < thresh_pos^2
            wxstar[ii] += wx0[i]
        end
    end
    @assert isapprox(sum(wxstar), 1.0; atol=0.0001)

    wxstar, xstar, mxstar
end

end
