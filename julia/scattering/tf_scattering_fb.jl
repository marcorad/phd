module ScatteringTF

include("util.jl")
include("morlet.jl")
include("1d_scattering_fb.jl")


using .morlet
using .Scattering1D

"""
Stores the information required to sample from a time and frequency morlet.

# Fields
    - morlet_time: The time morlet operation on t
    - morlet_quefrequency: The frequency morlet operating on log(λ)
"""
mutable struct MorletConfig2D
    morlet_time::Scattering1D.MorletConfig
    morlet_quefrequency::Scattering1D.MorletConfig
end

struct MorletFBConfigTF
    scalogram_config::Union{Scattering1D.MorletFBConfig, Nothing}
    time_config::Scattering1D.MorletFBConfig
    quefrequency_config::Scattering1D.MorletFBConfig
end

function configure_filterbank_2d(Q::Number, Q_time::Number, Q_quefrequency::Number, T_time::Number, T_quefrequency::Number, fs::Number, 
    ; fstart::Number = 0.0, fend::Number = fs, approximation_support::Number=5.0)
    scalogram_config = configure_filterbank_1d(Q, T_time, fs; 
        approximation_support=approximation_support, downsample_mode= maximum_uniform, fstart=fstart, fend=fend)
    time_config = configure_filterbank_1d(Q_time, T_time, scalogram_config.morlets[1].output_fs; 
        approximation_support=approximation_support, downsample_mode=maximum_uniform)
    quefrequency_config = configure_filterbank_1d(Q_quefrequency, T_quefrequency, 1.0; 
        approximation_support=approximation_support, downsample_mode=maximum_uniform)
end



end