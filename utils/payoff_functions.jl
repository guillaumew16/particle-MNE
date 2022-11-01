module PayoffFunctions

export random_fourier_function_1D, 
    separable_random_fourier_function_1D,
    discretize_1D,
    random_fourier_function,
    random_gaussian_matrix

using LinearAlgebra: dot
using Random

function random_fourier_function_1D(orderx::Int, ordery::Int; rng=MersenneTwister(1234))
    coeffs = randn(rng, ComplexF64, (2*orderx+1, 2*ordery+1)) # N(0,1) iid
    function gfun(x, y)::Real
        out = 0
        for k = -orderx:orderx, l = -ordery:ordery
            out += coeffs[k+orderx+1, l+ordery+1] * exp(2*pi*im * (k*x+l*y))
        end
        real(out)
    end
    return gfun
end

function discretize_1D(gfun, mx::Int, my::Int; xmin=0., xmax=1., ymin=0., ymax=1.)
    deltax = 1/mx
    deltay = 1/my
    gmat = Array{Float64}(undef, mx, my) # payoff matrix
    for i=1:mx, j=1:my
        # gmat[i,j] = gfun(i*deltax, j*deltay)
        gmat[i,j] = gfun(xmin + (xmax-xmin)*i*deltax, ymin + (ymax-ymin)*j*deltay)
    end
    return gmat
end

"""
Returns a function gfun(x, y) drawn from GP with no decay

Be careful to make dimx and dimy "hard-coded/literal (Val(3)) or already specified in the type-domain" when calling this function
See https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-value-type
"""
function random_fourier_function(::Val{dimx}, ::Val{dimy}, orderx::Int, ordery::Int; rng=MersenneTwister(1234)) where {dimx, dimy}
    coeffs = randn(rng, ComplexF64, (Tuple(2*orderx+1 for _=1:dimx)..., Tuple(2*ordery+1 for _=1:dimy)...)) # N(0,1) iid
    function gfun(x, y)::Real
        out = 0
        for kl in CartesianIndices(coeffs)
            kk = Tuple(kl)[1:dimx] .- 1 .- orderx
            ll = Tuple(kl)[dimx+1:dimx+dimy] .- 1 .- ordery
            out += coeffs[kl] * exp(2*pi*im * (dot(kk,x)+dot(ll,y)) )
        end
        real(out)
    end
    return gfun
end

function random_gaussian_matrix(mx::Int, my::Int; rng=MersenneTwister(1234), sigma=1.0)
    return sigma * rand(rng, Float64, mx, my)
end

end
