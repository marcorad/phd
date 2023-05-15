classdef Features

    properties(Constant)
        hrtol = 0.02;
        NHarmonics = 10;
        ratios = Features.getRatios();
        ratiosTolUp = Features.ratios*(1+Features.hrtol);
        ratiosTolDown = Features.ratios/(1+Features.hrtol);  
        tau = 10;
    end

    methods(Static)        

        function r = getRatios()
            t = 1:Features.NHarmonics;
            t = t./t';
            r = t(:);
            r = sort(r(r>1));
            r = unique(r)';
        end

        function x = load(feat)
            fname = POI.Path + feat.File;
            s = feat.Start;
            e = feat.End;
            x = audioread(fname, [s e]);
        end

        function plotQuintiles(Q, name)
            stem(Q)
            title(name)
            xlim([1,3])
            xticklabels(["min", "med", "max"])
        end

        function plot(feat)
            x = Features.load(feat);
            figure
            subplot(3,6,1:3)
            [S, f] = POI.PowerSpectrum(x);            
            t = (0:numel(x)-1)/POI.fs;
            Tools.plotTF(S, f, false, true, t);
            title(sprintf('%d Harmonics', feat.NHarmonics))
            subplot(3,6,4:6)
            plot(feat.FreqFeat)

            subplot(3, 6, 7)
            Features.plotQuintiles(feat.Freq, "Frequency");
            subplot(3, 6, 8)
            stem(feat.DeltaFreq)
            Features.plotQuintiles(feat.DeltaFreq, "Delta Frequency");
            subplot(3, 6, 9)
            Features.plotQuintiles(feat.Bandwidth, "Bandwidth");
            subplot(3, 6, 10)
            Features.plotQuintiles(feat.DeltaBandwidth, "Delta Bandwidth");
            subplot(3, 6, 11)
            Features.plotQuintiles(feat.PowerRatio, "Power Ratio");
            subplot(3, 6, 12)
            Features.plotQuintiles(feat.DeltaPowerRatio, "Delta Power Ratio");
        end

        function f = TimeRange(v)
            % f = prctile(v, [0,50,100]);
            m = median(v);
            md = median(abs(v - m));
            f = [m, md];
        end


        function [f, S, Smed, Sden, St] = Spectrograms(poi)          
            k1 = POI.k1;
            k2 = POI.k2;
            SEmask = poi.SignalProbability > 0.5;
            x = POI.loadPOI(poi);
            [S, f]= POI.PowerSpectrum(x);
            S = S(k1:k2, :);
            f = f(k1:k2);
            Smed = poi.MedianSpectrum;
            Smed = Smed(k1:k2, :);
            St = S.*SEmask >= Smed*Features.tau;
            Sden = max(S - Smed, 1e-12);
        end

        function Cdata = RemoveShortContours(Cdata, Nmin)
            remove = zeros(1, numel(Cdata)) == 1;
            for i = 1:numel(Cdata)
                c = Cdata(i);                
                dur = c.tmax - c.tmin + 1;
                if dur <= Nmin
                    remove(i) = true;
                end
                Cdata(i).NHarmonics = 0;
                Cdata(i).NContam = 0;
                Cdata(i).ContamFreq = [];
            end
            Cdata = Cdata(~remove);
        end

        function Cdata = RemoveHarmonics(Cdata)            
            remove = zeros(1, numel(Cdata)) == 1;    
            for i = 1:numel(Cdata)
                ci = Cdata(i);
                for j = 1:numel(Cdata)
                    cj = Cdata(j);
                    sameTime = ci.tmin <= cj.tmax & ci.tmax >= cj.tmin;
                    if sameTime && i ~= j
                        cijtmin = min(ci.tmin, cj.tmin);
                        cijtmax = max(ci.tmax, cj.tmax);
                        ovl = min(ci.tmax, cj.tmax) - max(ci.tmin, cj.tmin) + 1;                        
                        f0i = NaN(cijtmax - cijtmin + 1, 1);
                        f0j = f0i;
                        f0i((ci.tmin:ci.tmax) - cijtmin + 1) = ci.f0;
                        f0j((cj.tmin:cj.tmax) - cijtmin + 1) = cj.f0;
                        hratio = f0j./f0i;
                        tols = (hratio <= Features.ratiosTolUp & hratio >= Features.ratiosTolDown);
                        withinTol = sum(tols, 1);
                        withinTol = any(withinTol >= ovl*POI.PHarmonic);
                        dur = max(cj.tmax - cj.tmin, ci.tmax - ci.tmin);
%                         sameDur = ovl/dur >= (POI.PDuration);
%                         higherFreq = sum(hratio > 1, "all") > 0;
                        if withinTol
                            remove(j) = true;
                            if mean(abs(hratio - round(hratio))) < 0.05
                                Cdata(i).NHarmonics = Cdata(i).NHarmonics + 1;    
                            else
                                Cdata(i).NContam = Cdata(i).NContam + 1;
                                Cdata(i).ContamFreq = [Cdata(i).ContamFreq, median(cj.f0)];
                            end
                        else
                            Cdata(i).NContam = Cdata(i).NContam + 1;
                            Cdata(i).ContamFreq = [Cdata(i).ContamFreq, median(cj.f0)];
                        end
                    end
                end
            end

            
            Cdata = Cdata(~remove);
        end

        function dx = Delta(x)
            dt = (POI.Nfft - POI.Noverlap)/POI.fs;
            dx = diff(x)/dt;
        end

        function cfeats = Construct(Cdata, poi, St, Sden, S)
                ds = POI.getDataset(convertStringsToChars(poi.File));
                cfeats = {};
                poih = poi.H;
                fs = POI.fs;
                Nfft = POI.Nfft;
                FBpower = POI.MFCCFilterBank(:, POI.k1:POI.k2).^2;  
                k1 = POI.k1;
                Noverlap = POI.Noverlap;
                for i = 1:numel(Cdata)
                    c = Cdata(i);
                    %Bandwidth
                    BW = Features.TimeRange(c.bw);
                    %Delta Bandwidth
                    dBW = Features.TimeRange(Features.Delta(c.bw));
                    %Freq
                    F0 = Features.TimeRange(c.f0);
                    %DeltaFreq
                    dF0 = Features.TimeRange(Features.Delta(c.f0));
                    %Power of Contour
                    P = Features.TimeRange(c.P);
                    %DeltaPower of Contour
                    dP = Features.TimeRange(Features.Delta(c.P));
                    %TotalPower
                    kbot = max(min(floor((c.f0 - c.bw/2)/fs*Nfft) - k1),1); %only consider above F0min - BWmax
                    mask = ~St(:,c.tmin:c.tmax);
                    Sm = Sden(:,c.tmin:c.tmax);
                    Sm(mask) = 0;
                    tp = sum(Sm(kbot:end, :), 1);
                    TotP = Features.TimeRange(tp);
                    dTotP = Features.TimeRange(Features.Delta(tp));
                    %PowerRatio
                    pr = c.P ./ tp;
                    PRatio = Features.TimeRange(pr);
                    dPRatio = Features.TimeRange(Features.Delta(pr));
                    %Entropy
                    h = poih(:,c.tmin:c.tmax);
                    entropy = Features.TimeRange(h);
                    dentropy = Features.TimeRange(Features.Delta(h));

                    Pnoise = max(sum(S(kbot:end, c.tmin:c.tmax), 1) - tp, 0);

                    SNR = mag2db(mean(tp./Pnoise));

                    Tmin = (c.tmin-1)*(Nfft - Noverlap);
                    Tdur = (c.tmax - c.tmin + 1)*(Nfft - Noverlap);

                    Sc = Sden(:, c.tmin:c.tmax);
                    Sc = mean(Sc, 2);
                    Sc = FBpower*Sc;
                    Sc = Sc./sum(Sc, 'all');

                    s = poi.StartIndex;
                    
                    cfeats{i} = struct('Duration', Tdur/fs,...
                        'FreqFeat', Sc', ...
                        'Bandwidth', BW, 'DeltaBandwidth', dBW, 'Freq', F0, ...
                        'DeltaFreq', dF0, 'Power', P, 'DeltaPower', dP, ...
                        'TotPower', TotP, 'DeltaTotPower', dTotP,...
                        'PowerRatio', PRatio, 'DeltaPowerRatio', dPRatio, ...
                        'Entropy', entropy, 'DeltaEntropy', dentropy, ...
                        'NHarmonics', c.NHarmonics, 'NContam', c.NContam, 'ContamMedFreq', c.ContamFreq, ...
                        'AllContaminationLower', all(F0(1) > c.ContamFreq), 'BLSNR', SNR, ...
                        'File', poi.File, 'Start', s + Tmin, 'End', s + Tmin + Tdur, 'Dataset', ds);
                end

        end

        function D = Create(poi, wtq)
            N = size(poi, 2);
            D = [];
            for n = 1:N                
                [f, S, ~, Sden, St] = Features.Spectrograms(poi(n));
                Cdata = Features.Contours(Sden, St, f);
                Cdata = Features.RemoveShortContours(Cdata, POI.Nmin);
                Cdata = Features.RemoveHarmonics(Cdata);
                cfeats = Features.Construct(Cdata, poi(n), St, Sden, S);
                if ~isempty(cfeats)
                    D =  [D,cfeats{:}];
                end
                if nargin > 1
                    send(wtq, []);
                end
            end
            D = struct2table(D);
        end

        function D = Contours(S, St, f)
            CC = bwconncomp(St, 8);
            Nfft = POI.Nfft;
            fs = POI.fs;
            D = {};
            for i = 1:CC.NumObjects
                idx = CC.PixelIdxList{i};
                [fi, ti] = ind2sub(size(St), idx);
                fmin = min(fi);
                fmax = max(fi);
                tmin = min(ti);
                tmax = max(ti);
                mask = zeros( fmax - fmin + 1, tmax - tmin + 1);
                maskamp = zeros( fmax - fmin + 1, tmax - tmin + 1);
                tishift = ti - tmin + 1;
                fishift = fi - fmin + 1;
                mask(sub2ind(size(mask), fishift, tishift)) = 1;
                maskamp(sub2ind(size(mask), fishift, tishift)) = S(idx);
                frange = 0:size(mask, 1)-1;
                ampnorm = maskamp./sum(maskamp, 1);
                cenfidx = sum(frange'.*ampnorm, 1);
                cenf = (cenfidx + fmin - 1)/Nfft * fs + f(1);
                p = sum(maskamp, 1);
                bw = sqrt(sum((frange'-cenfidx).^2.*ampnorm, 1))/2;
                cdata.f0 = cenf;
                cdata.P = p;
                cdata.tmin = tmin;
                cdata.tmax = tmax;
                cdata.fmin = fmin;
                cdata.fmax = fmax;
                cdata.bw = bw;
                cdata.idx = idx;
                D{i} = cdata;
            end
            D = cell2mat(D);
        end

        function [X, cols] = logPositives(X)
            cols = all(X > 0, 1);
            X(:,cols) = log(X(:, cols));
            cols = find(cols);
        end

        function [X, names, feats, logcols] = PrepareDataMatrix(feats)

            cols = ["Freq", "DeltaFreq", "Power", "DeltaPower", "TotPower",...
                "DeltaTotPower", "PowerRatio", "DeltaPowerRatio", ...
                "Bandwidth", "DeltaBandwidth", "Entropy", "DeltaEntropy", "FreqFeat", "Duration"];
            X = table2array(feats(:, cols));
            [X, idx] = POI.RemoveOutliers(X);
            logcols = [];
            X = (X-mean(X, 1))./std(X, 1);

            names = "";
            feats = feats(idx, :);

            for s = cols(1:12)
                names(end+1) = s + "Min";
                names(end+1) = s + "Q1";
                names(end+1) = s + "Median";
                names(end+1) = s + "Q3";
                names(end+1) = s + "Max";
            end
            for i=1:size(POI.MFCCFilterBank, 1)
                names(end+1) = "FB" + int2str(i);
            end
            names(end+1) = "Duration";
            names = names(2:end);

        end

        function [X, idx] = RemoveOutliers(X)
            t = POI.PrcOutliers;
            [X, idx] = rmoutliers(X,"percentiles", [t,100-t]);
            idx = ~idx;
        end

        function filt = FilterFeatures(feats, varargin)
            n = numel(varargin);
            filt = feats;
            for i = 1:3:n
                f = varargin{i};
                l = varargin{i+1};
                c = varargin{i+2};
                crit = filt{:, f};
                if size(crit, 2)  > 1
                    crit = crit(:, c);
                end
                idx =  crit >= l(1) & crit <= l(2);
                filt = filt(idx, :);
            end
        end

    end

end