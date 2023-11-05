include("morlet.jl")


using Plots
using FFTW

plotlyjs()


n = 5.0;
Q = 8.0;
sigma = 2/(2^(1/Q) - 1) 

fs = 1000.0
tends = Int64(floor(n*sigma*fs/2)*2 + 1)
ts = collect((-tends):tends);

t = ts ./ fs


mor = morlet.sample_morlet(t, 1.0, sigma)
mor2 = morlet.sample_morlet(t,  2^(1/Q), sigma)
mor3 = morlet.sample_morlet(t,  10*2^(2/Q), sigma)

p = plot(t, real(mor));
plot!(t, imag(mor));
plot!(t, abs.(mor));
display(p);

ω_s = fs * 2.0 * pi

ω = collect(0:(length(t)-1))./length(t) * ω_s

psi1 = abs.(fft(mor))
psi2 = abs.(fft(mor2))
psi3 = abs.(fft(mor3))
p2 = plot(ω, psi1);
plot!(ω, psi2);
plot!(ω, psi3);
plot!(ω, psi1 + psi2 + psi3);
# xlims!(0, 5)
display(p2)


