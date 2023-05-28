classdef Contours < handle

    properties(Constant)
        hrtol = 0.02;
        NHarmonics = 10;
        ratios = Features.getRatios();
        ratiosTolUp = Features.ratios*(1+Features.hrtol);
        ratiosTolDown = Features.ratios/(1+Features.hrtol);
        tau = 10;
    end

    properties

    end

    methods(Static)
        function r = getRatios()
            t = 1:Features.NHarmonics;
            t = t./t';
            r = t(:);
            r = sort(r(r>1));
            r = unique(r)';
        end
    end

    methods

        function obj = Countours()

        end

        


    end


end