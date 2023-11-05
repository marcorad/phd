folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
ann = load(folder + "\annotations\annotations.mat").annotationTable;

classes = ["Bm_Ant-A", "Bm_Ant-B", "Bm_Ant-Z", "Bm_D", "Bp_20Plus", "Bp_20Hz"];

for c = classes
    mkdir(sprintf("fig/vocalisations/%s", c));
end

filenum = [984, 387, 98, 98, 30];

groupcounts(ann, "Annotation")
fb = load(folder + "\scattering\filterbank.mat").fb;
fb1 = fb.filterBanks(1);
f = fb1.fc;
fs = fb1.getSSamplingFreq();

parfor c = 1:numel(classes)
    class = classes(c);
    ann_filter = ann(ann.Annotation == class, :);
    
    for i = 1:size(ann_filter, 1)
    a = ann_filter(i,:);
    fname = replace(a.File, ".wav", ".mat");
    s = load(folder + "\scattering\" + fname).s;
    s = s(1:numel(f), :);

    fig = figure;
    t = [a.ScatteringStartIndex, a.ScatteringEndIndex]/fs;
    t = t + [-1 1] * 5;
    Tools.plotTF(s,f, true, true, fb1.getTime(s) - t(1));    
    xlim(t - t(1))
    ylim([min(f), max(f)]);
    xlabel("Time (s)")
    saveas(fig, sprintf("fig/vocalisations/%s/%s_%d.jpg", class, class, i))
    close(fig);
    end

end