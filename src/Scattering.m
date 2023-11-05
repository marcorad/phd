classdef Scattering < handle

    properties
        filterBanks (:, 1) SFB
        paths
        pathsizes
        pathstart
    end

    methods
        function obj = Scattering(Q, T, fs, N, flow, fhigh, oversample, s2fstart)
            obj.filterBanks(1) = SFB(Q(1), T, fs, N, flow, fhigh, true, oversample);
            fsfb = fs/obj.filterBanks(1).downsampleU;
            Nfb  = obj.filterBanks(1).Nu;
            for i = 2:numel(Q)         
                prevfb = obj.filterBanks(i-1);
                if nargin < 7
                    s2fstart = 1/T; %if not specified, start where the lpf cuts off
                end
                obj.filterBanks(i) = SFB(Q(i), T, fsfb, Nfb, s2fstart, max(prevfb.psiBWHz), false, oversample);
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
            prevbw = fb.psiBWrad;
            path = {1};
            pathstartidx = 1;
            while currfb <= numel(obj.filterBanks)
                fb = obj.filterBanks(currfb);                
                for i = 1:numel(prevu)
                    u = prevu{i};
                    for j = 1:size(u, 1)
                        [s, unext] = fb.filterSU(u(j, :), prevbw(j));
                        curru{end + 1} = unext;
                        coeffs{end + 1} = s;
                        path{end+1} = [path{pathstartidx}, j];
                    end
                    pathstartidx = pathstartidx +1;
                end
                prevu = curru;
                currfb = currfb + 1;
                curru = {};
                prevbw = fb.psiBWHz;
            end
            nanidx = cellfun(@(c) numel(c)>1,coeffs);
            pathsize = cellfun(@(p) size(p, 1), coeffs);
            coeffs = vertcat(coeffs{nanidx});
            path = path(nanidx);
            pathsize = pathsize(nanidx);
            obj.paths = path;
            obj.pathsizes = pathsize;
            obj.pathstart = cumsum([1, obj.pathsizes(1:end-1)]);
            N = numel(x);
            Nspec = ceil(N/obj.filterBanks(1).downsampleU/obj.filterBanks(1).downsampleS);
            coeffs = coeffs(:, 1:Nspec);
        end
    end

end