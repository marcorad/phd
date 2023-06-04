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
        
    end

    methods(Static)
        function d = timedec(v)
            % f = prctile(v, [0,50,100]);
            m = median(v);
            md = median(abs(v - m));
            mx = max(v);
            mn = min(v);
            d = [m, md, mx-mn];
        end
    end

end