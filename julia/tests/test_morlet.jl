include("../1d_scattering_fb.jl")

using .Scattering1D
using FFTW
using Plots
plotlyjs()

fb::Scattering1D.MorletFBConfig = configure_filterbank(16, 1.5, 250.0; scaling_mode=linear_for_larger_time_support, downsample_mode=maximum_uniform, fstart = 10.0)

irs = create_1d_scattering_filters(fb)

IRS = fft(irs, 1:1)
plot(abs.(IRS))