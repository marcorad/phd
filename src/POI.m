classdef POI

    properties(Constant)
        Path = "D:\Whale Data\POIData\";
        OverlapFactor = 7/8;
        Nfft = 512;
        Noverlap = POI.Nfft * POI.OverlapFactor;
        f1 = 20;
        f2 = 1000;
        k1 = floor(POI.f1/POI.fs*POI.Nfft)+1;
        k2 = floor(POI.f2/POI.fs*POI.Nfft)+1;
        M = POI.Nmin*2-1;
        R = 1;
        L = 2^18;
        fs = 2000;
        GuassStd = 0.5;
        Nmin = floor(0.25/((POI.Nfft-POI.Noverlap)/POI.fs));
        FreqMF = 15;
        PHarmonic = 0.8;
        PDuration = 0.9;
        PrcOutliers = 0.01;
        SMedSmooth = 5;

        Dataset1Time = [datetime("20220816", "InputFormat","yyyyMMdd"),...
            datetime("20220926", "InputFormat","yyyyMMdd")];
        Dataset2Time = [datetime("20220930", "InputFormat","yyyyMMdd"),...
            datetime("20221107", "InputFormat","yyyyMMdd")];
        %Average power difference of 10% between datasets when measured
        %above 150HZ
        %         DatasetAvgPower = [1.66432975417193e-05, 1.49294258320771e-05];
        DatasetAvgPower = [1, 1]; %Avg power of the 2 sets

        MFCCFilterBank = designAuditoryFilterBank(POI.fs, ...
            "FFTLength",POI.Nfft, ...
            "NumBands",20, ...
            "FrequencyRange",[POI.f1, POI.f2]);

        MFCCNumCoeff = 20;
    end

    methods(Static)

        function t = filename2datetime(fname)
            fname = fname(1:(8+6+4+2));
            fname = convertCharsToStrings(fname);
            t = datetime(fname, "InputFormat","yyyyMMdd_HHmmss_SSSS");
        end

        function ds = getDataset(fname)
            t = POI.filename2datetime(fname);
            ds = 0;
            if t >= POI.Dataset1Time(1) && t <= POI.Dataset1Time(2)
                ds = 1;
            elseif t >= POI.Dataset2Time(1) && t <= POI.Dataset2Time(2)
                ds = 2;
            end
        end

        function n = time2psindex(t)
            n = floor(t*POI.fs/(POI.Nfft - POI.Noverlap));
        end

        function n = index2psindex(idx)
            n = floor(idx/(POI.Nfft - POI.Noverlap));
        end

        function P = DatasetPower(files)
            for i = 1:2
                idx = [files.Dataset] == i;
                dsfiles = files(idx);
                p = 0;
                N = size(dsfiles, 1);

                wb = waitbar(0);
                parforWaitbar(wb, N);
                wtq = parallel.pool.DataQueue;
                afterEach(wtq,@parforWaitbar);

                parfor j = 1:N
                    fname = convertCharsToStrings(dsfiles(j).name);
                    x = audioread(POI.Path + fname);
                    p = p + sum(x.^2)/numel(x)/N;
                    send(wtq, []);
                end
                P(i) = p;
                delete(wb);
            end
        end

        function [files] = DatasetInfo()
            files = dir(POI.Path);
            files = files(~[files.isdir]);
            for i = 1:size(files, 1)
                files(i).Dataset = POI.getDataset(files(i).name);
            end
        end

        function x = loadPOI(poi)
            fname = POI.Path + poi.File;
            s = poi.StartIndex;
            e = poi.EndIndex;
            x = audioread(fname, [s e])*poi.NormFactor;
        end

       

        function [S, f, t] = PowerSpectrum(x)
            Nfft = POI.Nfft;
            Noverlap = POI.Noverlap;
            w = blackman(Nfft);
            [S, f]= stft(x, POI.fs, "Window", w , "OverlapLength",Noverlap, "FFTLength",Nfft, "FrequencyRange","onesided");
            S = S.*conj(S)/POI.Nfft;
            t = (0:size(S, 2)-1)*(POI.Nfft - POI.Noverlap)/POI.fs;
        end

         

        function [x, S, f, t] = plotPOI(poi)
            x = POI.loadPOI(poi);
            [S, f, t] = POI.PowerSpectrum(x);
            figure
            subplot(211)
            Tools.plotTF(S, f, false, true, t);
            subplot(212)
            plot(t, poi.SignalProbability);
        end

        function Smed = NoiseEstimate(S)
            Smed = median(medfilt1(S, POI.FreqMF, [], 1, "omitnan","truncate"), 2);
            Smed = smoothdata(Smed, "gaussian", POI.SMedSmooth);
        end

        function res = FindPOI(type, ds, poi_idx, whiten)

            path = POI.Path;

            M = POI.M;
            f1 = POI.f1;
            f2 = POI.f2;
            R = POI.R;
            L = POI.L;

            files = POI.DatasetInfo();
            if nargin >= 2
                files = files([files.Dataset] == ds);
            end

            for i = 1:numel(files)
                f = files(i);
                name = f.name;
                names(i) = convertCharsToStrings(name);
                times(i) = datetime(convertCharsToStrings(name(1:end-4)), InputFormat="yyyyMMdd_HHmmss_SSSS");
            end

            if nargin <= 2
                N = numel(names);
                whiten = true;
            else
                N = numel(poi_idx);
                names = names(poi_idx);
                times = times(poi_idx);
                files = files(poi_idx);
            end

            wb = waitbar(0);
            parforWaitbar(wb, N);
            wtq = parallel.pool.DataQueue;
            afterEach(wtq,@parforWaitbar);

            fs = 2000;
            fs = fs/R;

            Nfft = POI.Nfft;
            k1 = POI.k1; %account for matlab indexing
            k2 = POI.k2;

            res = {};

            parfor i = 1:N
                warning('off','all')
                p = path + names(i);
                t = times(i);

                [x, ~] = audioread(p);
                norm = 1/POI.DatasetAvgPower(files(i).Dataset);
                x = x*norm;

                if R>1
                    x = resample(x, 1, R);
                end

                Nx = numel(x);
                n = 1;
                j = 1;
                reschunk = {}
                while n < Nx
                    e = n+L-1
                    if e > Nx
                        e = Nx;
                    end
                    S = POI.PowerSpectrum(x(n : e));
                    Smed = POI.NoiseEstimate(S);
                    if whiten
                        S = S./Smed;
                    end
                    probs = nan;
                    mu_ns= nan;
                    mu_s= nan;
                    sigma_ns= nan;
                    sigma_s= nan;
                    pi_ns= nan;
                    pi_s= nan;
                    converged= nan;
                    n_iter= nan;
                    LL= nan;
                    mon= nan;
                    fmin= nan;
                    fmax= nan;
                    Ht= nan;
                    contam= nan;
                    delta = nan;
                    H = nan;

                    if type == "kmeans"
                        [H, probs, mu_ns, mu_s, ~, delta, converged] = SignalDetection.softSignalDetectionIt(S(k1:k2, :), 0.999, M, 15);

                    elseif type == "gmm"
                        [H, probs, mu_ns, mu_s, sigma_ns, sigma_s, pi_ns, pi_s, converged, n_iter, LL, mon, fmin, fmax, Ht, contam] = SignalDetection.softSignalDetectionGMM(S(k1:k2, :), M);
                        delta = mu_ns - mu_s;
                    end
                    reschunk{j} = struct('StartTime', t + seconds((n-1)/fs), ...
                        'SignalMean', mu_s, 'NoSignalMean', mu_ns, 'MeanDelta', ...
                        delta, 'SignalStd', sigma_s, 'NoSignalStd', sigma_ns,'LogLikelihood', LL,...
                        'SignalProportion', pi_s, 'NoSignalProportion', pi_ns, 'Monotonic', mon,...
                        'MinProb', fmin, 'MaxProb', fmax, 'EntropyThreshold', Ht, 'ClassContamination', contam,'H',H,...
                        'Converged', converged, 'Iterations', n_iter, 'File', names(i), ...
                        'StartIndex', n*R, 'EndIndex', e*R, ...
                        'SignalProbability', probs, 'MedianSpectrum', Smed, 'NormFactor', norm, 'Dataset', files(i).Dataset);
                    n = n + L;
                    j = j + 1;
                end
                send(wtq, []);
                res{i} = reschunk;
            end

            RES = {};
            for i = 1:numel(res)
                r = res{i};
                for j = 1:numel(r)
                    c = r{j};
                    RES{end+1} = c;
                end
            end

            res = cell2mat(RES);
            [~, sortidx] = sort([res.StartTime]);
            res = res(sortidx);

            %             res = struct2table(res);

            delete(wb);

        end

        function [results, nmis] = CreateFeatures(poi)
            wb = waitbar(0);
            parforWaitbar(wb, numel(poi));
            wtq = parallel.pool.DataQueue;
            afterEach(wtq,@parforWaitbar);
            results = {};
            Np = 6;
            Nt = floor(numel(poi)/Np);
            parfor n = 1:Np
                s = (n-1)*Nt + 1;
                e = n*Nt;
                try
                    if e > numel(poi)
                        e = numel(poi);
                    end
                    results{n} = Features.Create(poi(s:e), wtq);
                catch E
                    fprintf("From %d to %d\n%s", s, e, getReport(E, 'extended'))
                end
            end
            results = vertcat(results{:});
            delete(wb)
            mask = any(ismissing(results, [NaN, inf, -inf]),2);
            nmis = sum(mask);
            results = results(~mask,:);

        end

        function sampleClustersWav(T, k, Nsamples, ds)
            path = POI.Path;
            dsstring = "dataset" + string(ds);
            mkdir("clustersamples/wav/", dsstring);
            mkdir("clustersamples/wav/" + dsstring + "/","all");
            TClust = {};
            for c = 1:k
                TClust{c} = T(T.Cluster == c, :);
            end
            parfor c = 1:k
                Tc = TClust{c};
                Nt = size(Tc, 1);
                if Nt ~= 0
                    Ns = min(Nsamples, Nt);
                    plotidx = randperm(Nt, Ns);
                    X = [];
                    mkdir("clustersamples/wav/" + dsstring + "/",string(c));
                    for i = 1:Ns
                        idx = plotidx(i);
                        ti = Tc(idx, :);
                        % Tc(idx, "File")
                        fname = path+ti.File;

                        info = audioinfo(fname);
                        s = ti.Start - 200;
                        e = ti.End + 200;
                        if s < 1
                            s = 1;
                        end
                        if e > info.TotalSamples
                            e = info.TotalSamples;
                        end
                        [x , fs]= audioread(fname, [s, e]);
                        [S, f]= POI.PowerSpectrum(x);
                        t = (0:size(S, 2)-1)/fs*(POI.Nfft - POI.Noverlap);
                        fig = figure;
                        set(gcf,'Visible','off')
                        Tools.plotTF(S, f, false, true, t)
                        title(sprintf("f0 = %.1f Hz, Dur = %.2f s", ti.Freq(2), ti.Duration))
                        saveas(fig, sprintf("clustersamples/wav/%s/%d/%d.png", dsstring, c, i));
                        xnorm = (x-mean(x, 1));
                        xnorm = xnorm/max(abs(xnorm));
                        audiowrite(sprintf("clustersamples/wav/%s/%d/%d.wav", dsstring, c, i), xnorm, POI.fs);
                        X = [X; xnorm/2];
                        close(fig);
                    end
                    fig = figure;
                    set(gcf,'Visible','off')
                    fig.Position = [0 0 1800 900]
                    [S, f]= POI.PowerSpectrum(X);
                    t = (0:size(S, 2)-1)/fs*(POI.Nfft - POI.Noverlap);
                    Tools.plotTF(S, f, false, true, t)
                    title(sprintf("All sampled snippets (%d)", size(Tc,1)))
                    saveas(fig, sprintf("clustersamples/wav/%s/%d/ALL.png", dsstring, c));
                    saveas(fig, sprintf("clustersamples/wav/%s/all/%d.png", dsstring, c));
                    close(fig);
                    audiowrite(sprintf("clustersamples/wav/%s/%d/ALL.wav", dsstring, c), X, POI.fs);
                    audiowrite(sprintf("clustersamples/wav/%s/all/%d.wav", dsstring, c), X, POI.fs);
                end
            end

        end

        function poi = DiscardPOIData(poi, p, plt)
            poi = poi([poi.Converged]);
            % figure
            if plt
                figure
                histogram([poi.MeanDelta])
                xlabel("\mu_{ns} - \mu_s")
                ylabel("Count")
                title("SE Class Mean Difference")
            end
            t = prctile([poi.MeanDelta], p);
            poi = poi([poi.MeanDelta] >= t);
        end

        function poi = Detect(ds)
            dsstring = sprintf("DS%d", ds);
            path = "store\";
            fname = path + "POI" + dsstring + ".mat";
            fname30 = path + "POI30" + dsstring + ".mat";
            if ~isfile(fname30)
                if ~isfile(fname)
                    fprintf("File %s not found. Performing detection.\n", fname)
                    poi = POI.FindPOI("gmm", ds);
                    save(fname, 'poi', '-v7.3');
                else
                    fprintf("File %s found. Loading detections.\n", fname)
                    load(fname)
                end
            end                

            if ~isfile(fname30)
                fprintf("File %s not found. Performing discard.\n", fname30)
                load(fname);
                poi = POI.DiscardPOIData(poi, 70, true);
                save(path + "POI30" + dsstring + ".mat", 'poi', '-v7.3');
            else
                fprintf("File %s found. Loading non-discarded detections.\n", fname30)
                load(fname30);
            end

        end

        function [feats, Xw] = Features(poi, ds)
            dsstring = sprintf("DS%d", ds);
            path = "store\";
            fname30 = path + "Features30" + dsstring + ".mat";
            if ~isfile(fname30)
                fprintf("File %s not found. Performing feature creation.\n", fname30)
                feats = POI.CreateFeatures(poi);
                [X, ~, feats, ~] = Features.PrepareDataMatrix(feats);
                r = rank(X);
                if r < size(X, 2)
                    fprintf("Non-full rank. Performing SVD.\n")
                    [~, ~, V] = svd(X', 'econ');
                    Vr = V(:, 1:r);
                    Xw = Vr*sqrt(size(X,1));
                else
                    fprintf("Full rank.\n")
                    Xw = X;
                end
                save(fname30, "feats", "Xw", '-v7.3');
            else
                fprintf("File %s found. Loading features.\n", fname30)
                load(fname30);
            end
        end

        function gmm = Cluster(Xw, K, ds)
            dsstring = sprintf("DS%d", ds);
            path = "store\";
            fname = path + "GMM" + dsstring + ".mat";
            if ~isfile(fname)
                fprintf("File %s not found. Training GMM.\n", fname)
                gmm = fitgmdist(Xw, K, "CovarianceType","diagonal", ...
                    "RegularizationValue", 0.001, "Start", "plus", "Replicates", 1, "SharedCovariance", true,...
                    'Options',statset('Display','iter','MaxIter',100,'TolFun',1e-5));
                save(fname, "gmm", '-v7.3');
            else
                fprintf("File %s found. Loading GMM.\n", fname)
                load(fname);
            end
        end

        function ProcessDataset(ds, K, T)
            poi = POI.Detect(ds);
            [feats, Xw] = POI.Features(poi, ds);
            gmm = POI.Cluster(Xw, K, ds);
            certainFeats = POI.CertainFeatures(feats, Xw, gmm, T, ds);
            fprintf("Sampling clusters.")
            POI.sampleClustersWav(certainFeats, K, 100, ds);
        end

        function certainFeats = CertainFeatures(feats, Xw, gmm, T, ds)
            dsstring = sprintf("DS%d", ds);
            path = "store\";
            fname = path + "CertainFeatures30" + dsstring + ".mat";
            if ~isfile(fname)
                fprintf("File %s not found. Finding snippets with posterior >= %d.\n", fname, T)
                post = gmm.posterior(Xw);
                certain_idx = any(post >= T, 2);
                [~, clust] = max(post, [], 2);
                certainFeats = feats;
                certainFeats.Cluster = clust;
                certainFeats(~certain_idx, :) = [];
                save(fname, "certainFeats", '-v7.3');
                save(path + "Posterior30" + dsstring + ".mat", "post", '-v7.3');
            else
                printf("File %s found. Loading snippets with high posterior.\n", fname)
                load(fname);
                load(path + "Posterior30" + dsstring + ".mat");
            end
        end





    end

end