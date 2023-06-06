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
    [T, featnames] = ContourFeatures.featureTable(D);
    catch err
        disp(getReport(err, "extended"))
    end
end
T = vertcat(T);
D = D{:};
featnames = featnames{1};
save(sprintf("store\\features_%d.mat", ds), "D", '-mat');
% T = ContourFeatures.featureTable(D);
save(sprintf("store\\feature_table_%d.mat", ds), "T", "featnames", '-mat');
end
end