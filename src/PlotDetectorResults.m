folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
dsname = "CaseyIslands2017";
fileidx = 9:748; %first files are clipping and corrupt

det = Detector(dsname);
det.detections();

detectors = load(folder + "\\detectors\\detectors.mat").D;
detectors = detectors(fileidx);
figw = 350;
figh = 280;

sizef = 48;

labels = [];
pkmeans = [];
psegmms1white = [];
psegmms1 = [];
psegmmscatwhite = [];
psegmmscat = [];
ebledwhite = [];
ebled = [];
downsample = 1;


for i = 1:numel(detectors)
    d = detectors(i);
    kmeans = d.kmeans;
    segmms1white = d.segmms1white;
    segmms1 = d.segmms1;

    bledwhite = d.bledwhite;
    bled = d.bled;

    if ~all(d.truelabels == 0)   
        psegmms1white = [psegmms1white, segmms1white.probs(1:downsample:end) * segmms1white.converged];
        psegmms1 = [psegmms1, segmms1.probs(1:downsample:end) * segmms1.converged];
        pkmeans = [pkmeans, kmeans.probs(1:downsample:end) * kmeans.converged];

        ebledwhite = [ebledwhite, bledwhite.probs(1:downsample:end)];
        ebled = [ebled, bled.probs(1:downsample:end)];
        labels = [labels, d.truelabels(1:downsample:end)];
    end
end

xc = 'fpr';
yc = 'tpr';
xlims = [0, 0.5];
ylims = [0, 1];

% idx = labels == 1;

figure
hold on
[x, y] = perfcurve(labels, pkmeans, 1, 'XCrit',xc, 'YCrit',yc);
plot(x, y)
[x, y] = perfcurve(labels, psegmms1white, 1, 'XCrit',xc, 'YCrit',yc);
plot(x, y)
[x, y] = perfcurve(labels, psegmms1, 1, 'XCrit',xc, 'YCrit',yc);
plot(x, y)
[x, y] = perfcurve(labels, ebledwhite, 1, 'XCrit',xc, 'YCrit',yc);
plot(x, y, '--')
[x, y] = perfcurve(labels, ebled, 1, 'XCrit',xc, 'YCrit',yc);
plot(x, y, '--')

hold off
legend("SE K-means", "SE GMM White", "SE GMM", "BLED White", "BLED")
xlabel("False Positive Rate")
ylabel("True Positive Rate")
xlim(xlims);
% ylim(ylims);
set(gcf, "Position", [0 0 figw figh])

hassignal = cellfun(@(y) any(y), {detectors.truelabels});
names = ["kmeans", "segmms1white", "segmms1", "bledwhite", "bled"];
%get the stats of non-converged files that contain signal
for name = names(1:3)
    name
    conv = arrayfun(@(r) r.(name).converged, detectors);
    notconvsignal = hassignal & ~conv;
    numnotconv = sum(notconvsignal)
    pnotconv = sum(notconvsignal)/sum(hassignal)
    nummissedcalls = sum(arrayfun(@(r) size(r.annotations, 1), detectors(notconvsignal)))
    totalcalls = sum(arrayfun(@(r) size(r.annotations, 1), detectors))
    pmissedcalls = nummissedcalls/totalcalls
end

rocdetectors = detectors;%(hassignal);
thresh.segmms1white = linspace(0,1, 20);
thresh.segmms1 = thresh.segmms1white;
thresh.kmeans = [linspace(0,0.9,20), 0.95, 0.99, 0.991, 0.999,1];
thresh.bledwhite = linspace(1,8, 20);
thresh.bled = linspace(1,10, 20);

for name = names(1:5)
    name
    det = [rocdetectors.(name)];    
    probs = {det.probs};    
    if name == "bled"
    probs = cellfun(@(p) p / min(p), probs, 'UniformOutput',false);
    end
    if name == "segmms1" || name == "segmms1white" || name == "kmeans"
        conv = arrayfun(@(d) d.converged, det);
        probs = arrayfun(@(i) probs{i} * conv(i), 1:numel(probs), 'UniformOutput',false);
    end
    anns = {rocdetectors.annotations};
    labels = {rocdetectors.truelabels};
    numdet.(name) = zeros(size(thresh));
    numnoisesamples.(name) = zeros(size(thresh));
    
    for k = 1:numel(thresh.(name))
        r = thresh.(name)(k);
        mask = cellfun(@(p) p > r, probs, 'UniformOutput',false);
        idx = 1:numel(mask);
        ndet = sum(arrayfun(@(i) numdetections(mask{i}, anns{i}), idx));
        nnoise = sum(arrayfun(@(i) noisesamplesdet(mask{i}, labels{i}), idx));
        numdet.(name)(k) = ndet;
        numnoisesamples.(name)(k) = nnoise;
    end
end

nann = sum(arrayfun(@(r) sum(contains(r.annotations.Annotation, "Bm") | contains(r.annotations.Annotation, "Bp")), rocdetectors));
nnoise = sum(arrayfun(@(r) sum(~r.truelabels), rocdetectors));

figure
hold on
for name = names(1:3)
    tpr = numdet.(name)/nann;
    pnoise = numnoisesamples.(name)/nnoise;    
    subplot(1, 2, 1)
    hold on
    plot(thresh.(name), tpr)
    xlabel("Threshold")
    ylabel("Detected calls")
    subplot(1, 2, 2)
    hold on
    plot(thresh.(name), pnoise)
    xlabel("Threshold")
    ylabel("Detected noise samples")    
end
hold off
legend("SE K-means", "SE GMM White", "SE GMM")
set(gcf, "Position", [0 0 figw figh])

figure
hold on
for name = names(1:5)
    tpr = numdet.(name)/nann;
    pnoise = numnoisesamples.(name)/nnoise;
    if contains(name, "bled")
        plot(pnoise, tpr, "--")
    else
        plot(pnoise, tpr)
    end
    
end
hold off
legend("SE K-means", "SE GMM White", "SE GMM", "BLED White", "BLED")
ylabel("Detected annotated calls")
xlabel("Detected noise samples")
xlim([0 0.5])
set(gcf, "Position", [0 0 figw figh])

function num = numdetections(mask, annotations)
    num = 0;
    for i = 1:size(annotations, 1)
        a = annotations(i, :);
        if any(mask(a.ScatteringStartIndex : min(a.ScatteringEndIndex, numel(mask)))) && (contains(a.Annotation, "Bm") || contains(a.Annotation, "Bp"))
            num  = num + 1;
        end
    end
end

function num = nummisses(mask, annotations)
    num = 0;
    for i = 1:size(annotations, 1)
        a = annotations(i, :);
        if ~any(mask(a.SpectrogramStartIndex : a.SpectrogramEndIndex))
            num  = num + 1;
        end
    end
end

function num = noisesamplesdet(mask, truelabels)    
%     falserle = Tools.rle(~truelabels);
%     num = 0;
%     for i = 1:size(falserle, 1)
%         a = falserle(i, :);
%         if any(mask(a.Start : a.End-1))
%             num  = num + 1;
%         end
%     end
      mask = mask & (~truelabels);
      num = sum(mask);
end