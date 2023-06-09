classdef Scattering < handle

    properties
        filterBanks %SFB filterbanks        
    end

    methods
        function obj = Scattering(Q, T, fs, N, flow, fhigh)
            fsfb = fs;
            Nfb = N;
            for i = 1:numel(Q)
                obj.filterBanks(i) = SFB(Q(i), T, fsfb, Nfb, flow(i), fhigh(i));
                fsfb = fsfb / obj.filterBanks(i).downsampleS;
                Nfb  = size(obj.filterBanks(i).Phi, 2);
            end
        end
    end

end