folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
fb = load(folder + "\scattering\filterbank.mat").fb;
fb = fb.filterBanks(1);
f = fb.fc;
l = fb.lambdas';


% dl = DataLoader2(folder + "\scattering", "mat", "parallel", true);
% 
% 
% mu = {};
% stdev = {};
% ts = {};
% 
% spmd
%     while ~dl.isComplete(spmdIndex)
%         [s, info, fs] = dl.next(spmdIndex);
%         ssq = s.^2;
%         s1 = s(1:numel(f), :); 
%         s1sq = ssq(1:numel(f), :);
%         t = info.time;
%         ts{end + 1} = t;
%         s1smooth = smoothdata(s1, 1, "movmedian", 9);
%         s1smooth = smoothdata(s1smooth, 2, "movmedian", floor(fs*5));
%         snoise = smoothdata(s1smooth, 2, "movmean", floor(fs*60));
%         sn = 20*log(abs(snoise)./sqrt(l));
%         snmu = mean(sn, 2);
%         snstd = std(sn, [], 2);
%     
%         mu{end + 1} = snmu;
%         stdev{end + 1} = snstd;
%     end
% end
% 
% mu = [mu{:}];
% stdev = [stdev{:}];
% ts = [ts{:}];
% 
% mu = horzcat(mu{:});
% stdev = horzcat(stdev{:});
% ts = horzcat(ts{:});

figure
mup = mu(:, (9:end));
tp = ts(9:end);
mun = mup - max(mup, [], "all");
surface(tp, f, mun, "EdgeColor", "none", "FaceColor","interp")
view(-45, 55);
% xlabel("Time")
ylabel("Frequency (Hz)")
zlabel("Normalised Power (dB)")

figure
stdp = stdev(:, (9:end));
tp = ts(9:end);
surface(tp, f, stdp, "EdgeColor", "none", "FaceColor","interp")
view(-45, 45);
% xlabel("Time")
ylabel("Frequency (Hz)")
zlabel("\sigma (dB)")

figure
avg_mup = mean(mun, 2);
max_mup = max(mun, [], 2);
min_mup = min(mun, [], 2);
avg_std = mean(stdp, 2);
hold on
plot(f, avg_mup)
plot(f, max_mup, 'k--');
plot(f, min_mup, 'k--');
hold off
xlabel("Frequency (Hz)");
ylabel("Normalised Power (dB)")
legend("Average Noise Profile", "Max and Min Bounds")
xlim([min(f), max(f)]);