folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
fb = load(folder + "\scattering\filterbank.mat").fb;
fb = fb.filterBanks(1);
f = fb.fc;
l = fb.lambdas';
dl = DataLoader2(folder + "\scattering", "mat", "single", true);
fs = fb.getSSamplingFreq();

idx = 105;
fname = dl.filelist(idx).name;

s = load(folder + "\scattering\" + fname).s;

ssq = s.^2;
s1 = s(1:numel(f), :); 
s1sq = ssq(1:numel(f), :);
s1smooth = smoothdata(s1, 1, "movmedian", 9);
s1smooth = smoothdata(s1smooth, 2, "movmedian", floor(fs*5));
snoise = smoothdata(s1smooth, 2, "movmean", floor(fs*60));
t = fb.getTime(s);

clims = log([min(s1, [], "all"), max(s1, [], "all")/2]);

figure
ax1 = subplot(121);
Tools.plotTF(log(s1), f, true, false, t)
clim(clims)
xlabel("Time (s)")
ylabel("Frequency (Hz)")
ax2 = subplot(122);
Tools.plotTF(log(snoise), f, true, false, t)
clim(clims)
xlabel("Time (s)")
ylabel("Frequency (Hz)")

linkaxes([ax1, ax2], 'xy');
ylim([min(f) max(f)])