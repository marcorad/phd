classdef ContourFeatures < handle

    properties
        FileID %file id of this contour
        ContourID %contour id of this contour
        Duration %Duration
        Spectrum %average spectra
        Bandwidth %bandwidth
        DeltaBandwidth %delta bandwidth
        Frequency %frequency
        DeltaFrequency %delta freq
        Power %contour power
        DeltaPower  %delta contour power
        TotalPower  %total power above contour
        DeltaTotalPower %delta total power above contour
        PowerRatio  %power ratio Power/TotalPower
        DeltaPowerRatio %delta power ratio
        Entropy %entropy
        DeltaEntropy %delta entropy
        Harmonics %number of detected harmonics
        Contamination %number of other contaminating contours
        ContaminationIDs %the cids of all the contaminating contours
        ContaminationLower %whether all contaminating contours are lower than this one
        BLSNR %estimated snr bandlimited to fundamental
        StartIndex %start index in S
        EndIndex %end index in S
    end

    properties(Constant)
        filter_names = ["Bandwidth", "DeltaBandwidth", "Frequency", "DeltaFrequency", "Power", "DeltaPower", "TotalPower", "DeltaTotalPower", "PowerRatio", "DeltaPowerRatio", "Entropy", "DeltaEntropy"];
        filter_types = ["Median", "MAD", "Min", "Q1", "Q3", "Max", "Start", "End", "IQR"];
        standalone_filters = ["FileID", "Harmonics", "BLSNR", "Duration"];
        struct_names = ContourFeatures.structNames();
    end

    methods

        function calculate(obj, dt)
            if nargin < 2
                dt = 1;
            end

            dfeats = ["DeltaBandwidth", "DeltaFrequency", "DeltaPower", "DeltaTotalPower", "DeltaPowerRatio", "DeltaEntropy"];
            for feat = dfeats
                dfeat = extractAfter(feat,"Delta");
                obj.(feat) = diff(obj.(dfeat))/dt;
            end

            feats = ["Bandwidth", "DeltaBandwidth", "Frequency", "DeltaFrequency", "Power", "DeltaPower", "TotalPower", "DeltaTotalPower", "PowerRatio", "DeltaPowerRatio", "Entropy", "DeltaEntropy"];
            for feat = feats
                decimated = ContourFeatures.timedec(obj.(feat));
                obj.(feat) = decimated;
            end
        end



        function m = featureVector(obj)
            feats = ContourFeatures.getFeatureVectorNames();
            m = zeros(numel(feats), numel(obj.Frequency));
            for i = 1:numel(feats)
                feat = feats(i);
                m(i, :) = obj.(feat);
            end
            m = m(:);
        end

        function s = toStruct(obj)
            for name = ContourFeatures.struct_names
                s.(name) = obj.(name);
            end
        end
    end

    methods(Static)

        function d = timedec(v)
            q = prctile(v, [0,25,75,100]);
            m = median(v);
            md = median(abs(v - m));
            d = [m, md, q, v(1), v(end), q(3) - q(2)];
        end

        function D = featureMatrix(features)
            d = numel(features(1).featureVector());
            N = numel(features);
            D = zeros(d, N);
            for i = 1:N
                D(:, i) = features(i).featureVector();
            end
        end

        function s = structNames()
            mc = ?ContourFeatures;
            l = mc.PropertyList;
            s = "";
            for i = 1:numel(l)
                p = l(i);
                if p.Constant == 0
                    s(end+1) = p.Name;
                end
            end
            s = s(2:end);
        end

        function Frequency = filter(T, col, type, bounds, isabs)
            if nargin < 5
                isabs = false;
            end
            vals = T{:, col};
            if type ~= ""
                vals = vals(:, type == ContourFeatures.filter_types);
            end
            if isabs
                vals = abs(vals);
            end
            Frequency = T(vals >= bounds(1) & vals <= bounds(2), :);
        end

        function m = columnMinMax(T, col, type, p)
            if nargin < 4
                p = 0;
            end
            vals = T{:, col};
            if type ~= ""
                vals = vals(:, type == ContourFeatures.filter_types);
            end
            vals(isinf(vals)) = nan;
            m = [prctile(vals, p), prctile(vals, 100-p)];
        end

    end

end