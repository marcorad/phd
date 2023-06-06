classdef ContourFeatures < handle

    properties
        fid %file id of this contour
        cid %contour id of this contour
        duration %duration
        spectrum %average spectra
        bw %bandwidth
        dbw %delta bandwidth
        f %frequency
        df %delta freq
        p %contour power
        dp  %delta contour power
        tp  %total power above contour
        dtp %delta total power above contour
        pr  %power ratio p/tp
        dpr %delta power ratio
        h %entropy
        dh %delta entropy
        harm %number of detected harmonics
        ncontam %number of other contaminating contours
        contamids %the cids of all the contaminating contours
        contamlower %whether all contaminating contours are lower than this one
        blsnr %estimated snr bandlimited to fundamental
        sidx %start index in S
        eidx %end index in S
    end

    methods

        function calculate(obj, dt)
            if nargin < 2
                dt = 1;
            end

            dfeats = ["dbw", "df", "dp", "dtp", "dpr", "dh"];
            for feat = dfeats
                dfeat = extractAfter(feat,"d");
                obj.(feat) = diff(obj.(dfeat))/dt;
            end

            feats = ["bw", "dbw", "f", "df", "p", "dp", "tp", "dtp", "pr", "dpr", "h", "dh"];
            for feat = feats
                obj.(feat) = ContourFeatures.timedec(obj.(feat));
            end
        end



        function m = featureVector(obj)
            feats = ContourFeatures.getFeatureVectorNames();
            m = zeros(numel(feats), numel(obj.f));
            for i = 1:numel(feats)
                feat = feats(i);
                m(i, :) = obj.(feat);
            end
            m = m(:);
        end

    end

    methods(Static)
        function feats = getFeatureVectorNames()
            %             feats = ["bw", "dbw", "f", "df", "p", "dp", "tp", "dtp", "pr", "dpr", "h", "dh"];
            feats = ["duration", "bw", "dbw", "f", "df", "p", "dp", "tp", "dtp"];
        end

        function d = timedec(v)
            % f = prctile(v, [0,50,100]);
            m = median(v);
            md = median(abs(v - m));
            mx = max(v);
            mn = min(v);
            d = [m, md, mx-mn];
        end

        function D = featureMatrix(features)
            d = numel(features(1).featureVector());
            N = numel(features);
            D = zeros(d, N);
            for i = 1:N
                D(:, i) = features(i).featureVector();
            end
        end

        function T = featureTable(features)
            T = {};
            mmr = ["bw", "dbw", "f", "df", "p", "dp", "tp", "dtp"];
            mmr_names = ["Bandwidth", "DeltaBandwidth", "Frequency", "DeltaFrequency", "Power", "DeltaPower", "TotalPower", "DeltaTotalPower"];
            N = numel(features);
            h = waitbar(0, 'Please wait...');
            for i = 1:N
                %convert to med, mad, range
                if mod(i, 100) == 0
                waitbar(i/N, h);
                end
                f = features(i);
                for j = 1:numel(mmr)
                    name = mmr(j);
                    t_name = mmr_names(j);
                    s.(sprintf("%s%s", t_name, "Median")) = f.(name)(1);
                    s.(sprintf("%s%s", t_name, "MAD")) = f.(name)(2);
                    s.(sprintf("%s%s", t_name, "Range")) = f.(name)(3);
                end
                s.Duration = f.duration;
                s.Spectrum = f.spectrum;
                s.FileID = f.fid;
                s.ContourID = f.cid;
                s.ContamIDs = f.contamids;
                s.ContamLower = f.contamlower;
                s.BLSNR  = f.blsnr;
                s.StartIndex = f.sidx;
                s.EndIndex = f.eidx;
                T{end+1} = s;
            end
            T = struct2table(cell2mat(T));
            close(h);
        end

        function f = filter(T, col, bounds)
            vals = T.(col);
            f = T(vals >= bounds(1) & vals <= bounds(2), :);
        end

        function m = columnMinMax(T, col)
            vals = T.(col);
            m = [min(vals), max(vals)];
        end

    end

end