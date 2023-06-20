function GenFeatures()

for ds = [1,2]
    load(sprintf("store\\poi_%d.mat", ds))
    dl = DataLoader(0);
    dl.auxLoader(infolist, "single");
    dl.resetIdx("parallel");
    dl.startAuxWaitbar();
    spmd
        D = [];
        try
            while ~dl.isAuxComplete(spmdIndex)
                [s, info] = dl.nextAux(spmdIndex);
                ct = Contours(dl.fb, s.^2, info);
                ct.fitGMM(true);
                ct.extractContours();
                D = [D, ct.features];
            end
        catch err
            disp(getReport(err, "extended"))
        end
    end
    fprintf("FINISHED DS %d SPMD\n", ds);
    D = D{:};   
    T = {};
    h = waitbar(0);
    N = numel(T);
    c = 0;
    for cf = D
        T{end+1} = cf.toStruct();
        c = c+1;
        if mod(c, 100) == 0
            waitbar(c/N, h);
        end
    end
    close(h);
    fprintf("CONVERTED DS %d TO TABLE\n", ds);
    T = struct2table([T{:}]);
    save(sprintf("store\\feature_table_%d.mat", ds), "T", '-mat');
    save(sprintf("store\\features_%d.mat", ds), "D", '-mat');
end
end