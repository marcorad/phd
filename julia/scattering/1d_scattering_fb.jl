
module Scattering1D

include("util.jl")
include("morlet.jl")

using .morlet

export configure_filterbank_1d, create_1d_scattering_filters

export _2sigma_at_next_morlet, _1sigma_at_next_morlet, equal_to_Q, equal_to_2Q
@enum MorletBW begin
    _2sigma_at_next_morlet
    _1sigma_at_next_morlet
    equal_to_Q
    equal_to_2Q
end

export linear_for_larger_time_support, always_exponential
@enum LambdasScalingMode begin
    linear_for_larger_time_support
    always_exponential
end

export off, maximum_split, maximum_uniform, multiple_of_lpf
@enum DownsampleMode begin
    off
    maximum_uniform
    maximum_split
    multiple_of_lpf
end

"""
Stores the information required to sample from a dilated morlet.

# Fields
    - λ: scaling factor resulting from this wavelet
    - f_c: center frequency (Hz)
    - ψ_σ_t: mother wavelet time support σ_t
    - σ_t: time support of this wavelet
    - σ_f: standard deviation in frequency (Hz)
    - σ_ω: standard deviation in frequency (rad/s)
    - is_linear: whether this filter is linearly spaced
    - BW_hz: the bandwidth of this wavelet (Hz), measured as 2*σ_f
    - downsampling_factor: the amount the wavelet may be downsampled as a result of its limited BW
    - output_fs: the resulting sampling frequency after this morlet has downsampled
"""
mutable struct MorletConfig
    λ::f64
    f_c::f64
    ψ_σ_t::f64 
    σ_t::f64 
    σ_f::f64
    σ_ω::f64 
    is_linear::bool
    BW_hz::f64 
    downsampling_factor::int32
    output_fs::f64
end

function to_str(mmd::MorletConfig)
    "
    λ: $(mmd.λ)
    f_c: $(mmd.f_c)
    ψ_σ_t: $(mmd.ψ_σ_t)
    σ_t: $(mmd.σ_t)
    σ_f: $(mmd.σ_f)
    σ_ω: $(mmd.σ_ω)
    is_linear: $(mmd.is_linear)
    BW_hz: $(mmd.BW_hz)
    downsampling_factor: $(mmd.downsampling_factor)
    output_fs: $(mmd.output_fs)
    "
end

# printing overrides to visualise the morlets
Base.print(io, mmd::MorletConfig) = print(io, to_str(mmd))
Base.print(mmd::MorletConfig) = print(to_str(mmd))

"""
Stores information required to construct 1D morlet filterbank.

# Fields
    - Q: Wavelets per octave
    - T: LPF time support
    - fs: Sampling frequency (Hz)
    - fstart: Starting frequency (Hz)
    - fend: Ending frequency (Hz)
    - Λ: Set of centre frequencies (rad/s) for convenience 
    - BW_mode: Sets the mother wavelet bandwidth, allowing for various widths to cover the spectrum
    - scaling_mode: Whether filters should be linearly packed, if applicable
    - morlets: Contains metadata about each morlet
    - downsample_map: Maps each downsampling factor to the indices of filters which may be downsampled by the amount    
    - number_of_wavelets: Number of filters in the filterbank
    - oversampling_factor: Oversampling factor set by the user
    - lpf_downsample_factor: The amount of dowmsampling required when unmodified scalogram is used
    - max_output_BW_hz: The maximum time-bandwidth resulting from the output scalogram
    - min_time_support_samples: The number of samples [-N,N] to fully approximate the filters based on their largest time support
    - approximation_support: The number of std. dev. in time to include in the filter approximations
"""
struct MorletFBConfig
    Q::int32
    T::f64
    fs::f64
    fstart::f64
    fend::f64
    Λ::Vector{f64}  
    BW_mode::MorletBW 
    scaling_mode::LambdasScalingMode 
    morlets::Vector{MorletConfig} 
    downsample_map::Dict{int32, Vector{int32}} 
    number_of_wavelets::int32 
    oversampling_factor::int32 
    lpf_downsample_factor::int32 
    max_output_BW_hz::f64 
    min_time_support_samples::int32 
    approximation_support::f64 
end

function lpf_downsampling(ωs, lpf_σ_ω, oversample)
    int32(floor(ωs/4/lpf_σ_ω/oversample))
end

function maximum_downsampling(ωs, bw_rad, lpf_σ_ω, oversample)
    final_lpf_downsample = lpf_downsampling(ωs, lpf_σ_ω, oversample)
    int32(floor(min(max(ωs/2/bw_rad/oversample, 1), final_lpf_downsample)))
end


function configure_filterbank_1d(Q::Number, T::Number, fs::Number; scaling_mode::LambdasScalingMode = linear_for_larger_time_support, BW_mode::MorletBW = _2sigma_at_next_morlet,
    fstart::Number=0.0, fend::Number=fs, approximation_support::Number=5.0, oversampling_factor::Integer = 1, downsample_mode::DownsampleMode=off)::MorletFBConfig

    #choose the BW of each wavelet according to the mode
    ψ_σ_ω = 0.0
    if BW_mode == _2sigma_at_next_morlet
        ψ_σ_ω = (2^(1 / Q) - 1) / 2
    elseif BW_mode == _1sigma_at_next_morlet
        ψ_σ_ω = (2^(1 / Q) - 1)
    elseif BW_mode == equal_to_Q
        ψ_σ_ω = 1 / Q
    elseif BW_mode == equal_to_2Q
        ψ_σ_ω = 2 / Q
    end
    ψ_σ_t = 1/ψ_σ_ω

    ωs = 2π * fs

    #determine the time supports and starting frequency according to T, fstart, fend
    λ_0 = 2π * fstart
    λ_end = min(2π * fend, fs*2π)/2

    lpf_σ_ω = 2π / T #sets the cut-off frequency to 1/T Hz, with cut-off defined as 1 standard deviation in the frequency domain
    lpf_σ_t = 1 / lpf_σ_ω

    if λ_0 < 2*lpf_σ_ω
        λ_0 = 2*lpf_σ_ω
    end

    #start to assemble the centre frequencies
    Λ = Vector{f64}(undef, 0)
    morlets::Vector{MorletConfig} = Vector{MorletConfig}(undef, 0)

    λ = λ_0
    if scaling_mode == linear_for_larger_time_support

        bw::f64 = ψ_σ_ω * λ
        dλ::f64 = 2*lpf_σ_ω #TODO: set dλ so that similar placement is achieved compared to the exponentially scaled filters

        while bw < lpf_σ_ω
            push!(Λ, λ)
            ψ_σ_t_lin = lpf_σ_t * λ #ensures that the time support of the wavelet will result in a BW equal to lpf_σ_ω
            ψ_σ_ω_lin = 1 / ψ_σ_t_lin
            bw_lin = 2*ψ_σ_ω_lin*λ
            downsample = maximum_downsampling(ωs, bw_lin, lpf_σ_ω, oversampling_factor) 
            push!(morlets, MorletConfig(λ, λ/2π, ψ_σ_t_lin, ψ_σ_t_lin/λ, ψ_σ_ω_lin/2π, ψ_σ_ω_lin, true, bw_lin/2π, downsample, fs))
            λ += dλ
            bw = ψ_σ_ω * λ            
        end      

    end

    while λ < λ_end
        push!(Λ, λ)     
        bw = ψ_σ_ω * λ
        downsample = maximum_downsampling(ωs, 2*bw, lpf_σ_ω, oversampling_factor)   
        push!(morlets, MorletConfig(λ, λ/2π, ψ_σ_t, ψ_σ_t/λ, 1/ψ_σ_t/2π, 1/ψ_σ_t, false, 2/ψ_σ_t/2π*λ, downsample, fs))
        λ *= 2^(1/Q)
    end

    lpf_downsample = lpf_downsampling(ωs, lpf_σ_ω, oversampling_factor)
    #depending on the downsampling mode, modify the wavelets
    if downsample_mode == off
        for m in morlets
            m.downsampling_factor = 1
        end
    elseif downsample_mode == maximum_uniform
        ds = morlets[end].downsampling_factor
        for m in morlets
            m.downsampling_factor = ds
        end
    elseif downsample_mode == multiple_of_lpf        
        for m in morlets
            ds = m.downsampling_factor
            #find the closest downsampling factor that will allow for a further dowmsampling step to the LPF
            while lpf_downsample % ds != 0
                ds -= 1
            end
            m.downsampling_factor = ds
        end
    end

    #calculate the resulting sample frequency after each downsampling step
    for m in morlets
        m.output_fs = fs/m.downsampling_factor
        print(m)
    end

    #prepare the DS map
    map::Dict{int32, Vector{int32}} = Dict{int32, Vector{int32}}()

    for i in 1:length(morlets)
        m::MorletConfig = morlets[i]
        if !haskey(map, m.downsampling_factor)
            map[m.downsampling_factor] = [i]
        else
            append!(map[m.downsampling_factor], i)
        end
    end

    print(map)

    ns = int32(ceil(max(morlets[1].σ_t*fs*approximation_support, lpf_σ_t*approximation_support*fs)))
    
    MorletFBConfig(Q, T, fs, fstart, fend, Λ, BW_mode, scaling_mode, morlets, map, length(morlets), oversampling_factor, lpf_downsample, morlets[end].BW_hz, ns, approximation_support)

end

function create_1d_scattering_filters(fb::MorletFBConfig)
    impulse_responses = zeros(c64, fb.min_time_support_samples*2 + 1, fb.number_of_wavelets)
    ns = fb.min_time_support_samples
    t = collect((-ns):ns)./fb.fs
    for i in 1:length(fb.morlets)
        m::MorletConfig = fb.morlets[i]
        impulse_responses[:, i] = sample_morlet(t, m.λ, m.ψ_σ_t)
    end
    impulse_responses
end

end