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