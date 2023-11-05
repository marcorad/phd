include("../morlet.jl")

using .morlet
using Plots
using FFTW


fs = 32.0
lambda = 1.0

sigma = 2.0
approx_support = 3.0
N = sigma*approx_support*fs
t = collect(Float64, (-N):N)
t = t/fs

print(t)

mor2d = sample_morlet_2d(t, lambda*9, sigma, t, lambda*128, sigma)


p1 = heatmap(t, t, imag(mor2d))
display(p1)
heatmap(transpose(abs.(fft(mor2d, 1:2))))



# readline()

