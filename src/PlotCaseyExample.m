folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
fb = load(folder + "\scattering\filterbank.mat").fb;
fb = fb.filterBanks(1);
f = fb.fc;
l = fb.lambdas';
dl = DataLoader2(folder + "\scattering", "mat", "single", true);
fs = fb.getSSamplingFreq();

idx = 116;
fname = dl.filelist(idx).name;

s = load(folder + "\scattering\" + fname).s;

ssq = s.^2;
s1 = s(1:numel(f), :); 
s1sq = ssq(1:numel(f), :);
s1smooth = smoothdata(s1, 1, "movmedian", 9);
s1smooth = smoothdata(s1smooth, 2, "movmedian", floor(fs*5));
snoise = smoothdata(s1smooth, 2, "movmean", floor(fs*60));
t = fb.getTime(s);

ann = load(folder + "\annotations\annotations.mat").annotationTable;

ann = ann(ann.File == replace(fname, ".mat", ".wav"), :);

M = floor(Detector.Tmf * fs)*2 + 1;
segmm = SEGMM(M);
segmm.detect((s1./snoise).^2);

H = segmm.H;
PCs = segmm.probs; 

figure

ax1 = subplot(311);
hold on
Tools.plotTF(s1, f, true, true, t);
for i = 1:size(ann, 1)
    a = ann(i, :);
    ta = [a.ScatteringStartIndex, a.ScatteringEndIndex]/fs;
    fa = [a.StartFrequency, a.EndFrequency];
    fill([ta, fliplr(ta)], [fa(1), fa(1), fa(2), fa(2)], 'r-', 'FaceAlpha',0.2);
    text((ta(2)-ta(1))/2 + ta(1), fa(2), a.Annotation, "Interpreter","none", "HorizontalAlignment","center", VerticalAlignment="bottom")
end
ylim([min(f) max(f)])
hold off
ylabel("Frequency (Hz)")

ax2 = subplot(312);
plot(t, H);
ylabel("H_w")

ax3 = subplot(313);
plot(t, PCs);
xlabel("Time (s)");
ylabel("p(C_s | H_w)")


linkaxes([ax1 ax2], 'x')
linkaxes([ax1 ax3], 'x')