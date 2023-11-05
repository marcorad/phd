module morlet

include("util.jl")
export sample_morlet, sample_morlet_2d

const global j = 1im

function sample_gauss(t::f64, sigma::f64)::f64
    1.0 / sqrt(2.0 * pi) / sigma * exp(-0.5 * (t / sigma)^2)    
end

function sample_morlet(t::Vector{f64}, lambda::f64, sigma::f64)::Vector{c64}
    g = similar(t)
    g .= sample_gauss.(t .* lambda, sigma)
    beta = (sample_gauss(-1.0, 1.0 / sigma) / sample_gauss(0.0, 1.0 / sigma))
    lambda .* (exp.(j .* t .* lambda) .- beta) .* g
end

function sample_morlet_2d(t1::Vector{f64}, lambda1::f64, sigma1::f64, t2::Vector{f64}, lambda2::f64, sigma2::f64)
    sample_morlet(t1, lambda1, sigma1) * transpose(sample_morlet(t2, lambda2, sigma2))
end

end