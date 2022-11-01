module MiscUtils

export normalize!, torus_dist_1D, aggregate_particles_1D, linrescale,
    myCircle, mySphere, myDisk

using Random: MersenneTwister

function normalize!(a)
    s = sum(a)
    for i in eachindex(a)
        a[i] /= s
    end
end

torus_dist_1D(zi0, zi) = ( d = zi0 - zi ; min(abs(d), abs(d+1), abs(d-1)) )

"""
Given a reference point wx0,x0 encoded the same way as iterates i.e with mx particles, compute wxstar,xstar and mxstar such that nearby particles are aggregated into one.
Of course this only makes sense to do if wx0,x0 correponds to a sparse measure (sparser than mx).
    thresh_pos: two particles i, i' (in [mx]) are aggregated together if dist(xi, xi') < thresh

Returns wxstar, xstar, mxstar
"""
function aggregate_particles_1D(wx0, x0; thresh_pos=1e-5, torus=true)
    dist(z, z0, torus) = torus ? torus_dist_1D(z, z0) : abs(z-z0)

    xstar = similar(x0)
    mxstar = 0
    for i in eachindex(x0)
        duplicate = false
        for ii=1:mxstar
            if dist(xstar[ii], x0[i], torus) < thresh_pos
                duplicate = true
                break
            end
        end
        if !duplicate
            xstar[mxstar+1] = x0[i]
            mxstar += 1
        end
    end
    xstar = xstar[1:mxstar]
        
    wxstar = zeros(mxstar)
    for i in eachindex(x0), ii in eachindex(xstar)
        if dist(xstar[ii], x0[i], torus) < thresh_pos
            wxstar[ii] += wx0[i]
        end
    end
    @assert isapprox(sum(wxstar), 1.0; atol=0.0001)

    wxstar, xstar, mxstar
end

"""
Linearly rescale scalars from [xmin_from, xmax_from] to [xmin_to, xmax_to]
- clip=true: values falling on the left of xmin_from (resp right of xmax_from) are set to xmin_to (resp xmax_to)
- clip=false: values falling outside of the "from" domain are rescaled like the others
"""
function linrescale(x, xmin_from, xmax_from, xmin_to=0, xmax_to=1; clip=true)
    # rescale to [0,1]
    x = x .- xmin_from
    x = x ./ (xmax_from - xmin_from)
    # rescale to [xmin_to, xmax_to]
    x = x .* (xmax_to - xmin_to) 
    x = x .+ xmin_to
    # clip
    if clip
        x = max.(x, xmin_to)
        x = min.(x, xmax_to)
    end
    return x
end

"Generate N equidistant points on the unit circle"
function myCircle(N)
    z = exp.( 2*im*pi* range(0, 1, length=N+1)[1:end-1] )
    # real(z), imag(z)
    transpose(hcat(real(z), imag(z)))
end

"""
Generate at least N equidistant points on the unit sphere in dimension 3
Copied from: Brian Z Bentz (2022). mySphere(N) (https://www.mathworks.com/matlabcentral/fileexchange/57877-mysphere-n), MATLAB Central File Exchange. Retrieved July 21, 2022. 
    MATLAB function for generating equidistant points on the surface of a unit sphere.
"""
function mySphere(N)
    X = zeros(2N)
    Y = zeros(2N)
    Z = zeros(2N)
    r_unit = 1
    Area = 4*pi*r_unit^2/N
    Distance = sqrt(Area)
    M_theta = round(pi/Distance)
    d_theta = pi/M_theta
    d_phi = Area/d_theta
    N_new = 0
    for m=0:M_theta-1
        Theta = pi*(m+0.5)/M_theta
        M_phi = round(2*pi*sin(Theta)/d_phi) # not exact
        for n = 0:M_phi-1
            Phi = 2*pi*n/M_phi
            N_new = N_new + 1
            X[N_new]= sin(Theta)*cos(Phi)
            Y[N_new] = sin(Theta)*sin(Phi)
            Z[N_new] = cos(Theta)
        end
    end
    X = X[1:N_new]
    Y = Y[1:N_new]
    Z = Z[1:N_new]
    # X, Y, Z, N_new
    transpose(hcat(X, Y, Z)), N_new
end

"Generate at least N equidistant points from the unit ball"
function myBall(N, d=2)
    error(ErrorException("Not implemented"))
end

"""
Generate approximately N equidistant points from the unit disk
Adapted from: http://www.holoborodko.com/pavel/2015/07/23/generating-equidistant-points-on-unit-disk/
"""
function myDisk(N)
    if N<4
        throw(ArgumentError("N must be >=4"))
    end
    Nr = 1
    s = 6
    while s<N
        Nr += 1
        s += Int(round( pi/asin(1/(2*Nr)) ))
    end

    x = zeros(2N)
    y = zeros(2N)
    dR = 1/Nr
    x[1] = 0
    y[1] = 0
    k = 1
    idx = 1
    for r = dR:dR:1
        n = Int(round( pi/asin(1/(2*k)) ))
        theta = range(0, stop=2*pi, length=n+1)
        x[idx+1:idx+n] =  r.* cos.(theta[1:n])
        y[idx+1:idx+n] = r.* sin.(theta[1:n])
        k = k+1
        idx = idx+n
    end
    N_new = idx
    x = x[1:N_new]
    y = y[1:N_new]
    points = transpose(hcat(x, y))
    return points, N_new
end

"Generate N points from the unit disk fast"
function myDisk_fast(N; rng=MersenneTwister(1234))
    # phi01 = rand(rng, N)
    phi01 = range(0, 1, length=N+1)[1:end-1]
    r = rand(rng, N).^2
    z = r .* exp.( 2*im*pi* phi01 )
    transpose(hcat(real(z), imag(z)))
end

end
