function GenGMM(dl)
dl.resetIdx("parallel");
dl.startSpectrogramWaitbar()
spmd
    idx = 1;
    while ~dl.isSpectrogramComplete(spmdIndex)
        [S, info] = dl.nextSpectrogram(spmdIndex);
        try
        c = Contours(dl.fb.getTime(S), dl.fb.fc, S.^2, info);
        %first fit the whitened version
        c.fitGMM(true)
        gmm_white(idx) = c.getStatistics();
        c.fitGMM(false)
        gmm_not_white(idx) = c.getStatistics();
        idx = idx + 1;
        catch err
            fprintf("Error in file %s\n", info.name)
            disp(getReport(err, "extended"));                
        end
    end
end
gmm_white = [gmm_white{:}];
gmm_not_white = [gmm_not_white{:}];
save(sprintf("store\\gmm_%d.mat", dl.ds), "gmm_white", "gmm_not_white", '-mat');
end
