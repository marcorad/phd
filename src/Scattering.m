classdef Scattering < handle

    properties
        filterBanks (:, 1) SFB      
    end

    methods
        function obj = Scattering(Q, T, fs, N, flow, fhigh)
            fsfb = fs;
            Nfb = N;
            for i = 1:numel(Q)                
                obj.filterBanks(i) = SFB(Q(i), T, fsfb, Nfb, flow(i), fhigh(i), i == 1);
                fsfb = fsfb / obj.filterBanks(i).downsampleU;
                Nfb  = obj.filterBanks(i).Nu;
            end
        end

        function [coeffs, spec] = scattering(obj, x)
            coeffs = {};
            fb = obj.filterBanks(1);
            [spec, u] = fb.filterSU(x);
            coeffs{end+1} = spec;
            prevu = {u};
            curru = {};
            currfb = 2;
            while currfb <= numel(obj.filterBanks)
                fb = obj.filterBanks(currfb);
                for i = 1:numel(prevu)
                    u = prevu{i};
                    for j = 1:size(u, 1)
                        [s, unext] = fb.filterSU(u(j, :));
                        curru{end + 1} = unext;
                        coeffs{end + 1} = s;
                    end
                end
                prevu = curru;
                currfb = currfb + 1;
                curru = {};
            end
            coeffs = vertcat(coeffs{:});
            N = numel(x);
            Nspec = ceil(N/obj.filterBanks(1).downsampleU/obj.filterBanks(1).downsampleS);
            coeffs = coeffs(:, 1:Nspec);
        end
    end

end