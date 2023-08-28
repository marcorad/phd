classes = ["Bm_Ant_A", "Bm_Ant_B", "Bm_Ant_Z", "Bm_D", "Bp_20Plus"];
alphas = [0.69,0.69,0.48,0.69,0.48];
gammas = [0.33,0.55,0.55,0.78,0.55];

folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
data = load(folder + "\features\features.mat").data;

data.Annotation = replace(data.Annotation, "-", "_");

for i=1:numel(classes)

    class = classes(i)
    alpha = alphas(i);
    gamma = gammas(i);   

    %prepare the data
    X = data{:, "Features"};
    X = double(X);
    y = data{:, "Annotation"};
    
    %remove irrelevant classes
    idx = ~contains(y, "Multiple");
    idx = idx & all(~isnan(X), 2);
    y = y(idx);
    y(~contains(y, class)) = "Noise";

    X = X(idx, [1:end-1, end]);
    Xn = ((X-mean(X, 1))./std(X, 1));

    sigp = 1 - alpha;
    prior = struct("Bm_Ant_A",sigp,"Bm_Ant_B",sigp,"Bm_Ant_Z",sigp,"Bm_Ant",sigp, "Bm_D", sigp, "Bp_20Plus", sigp, "Noise", alpha);
    cls = Discriminant(gamma, "linear", 0);
    [Xtrain, ytrain, Xtest, ytest] = split(Xn, y, 0.8, 0.8);
    cls.fit(Xtrain, ytrain, prior);

    yhat = cls.predict(Xn);

    data(idx, sprintf("Predict_%s", class)) = table(yhat);

end

Nplot = 100;

files = unique(data.File);
idx = randperm(numel(files), Nplot);

fb = load(folder + "\scattering\filterbank.mat").fb;
fs = fb.filterBanks(1).getSSamplingFreq();
f = fb.filterBanks(1).fc;

ftext = f(end - [4, 8, 12, 16, 20 24]);

mkdir("detections");

for i = 1:Nplot
    fileidx = idx(i);
    fname = files(fileidx);
    s = load(folder + "\scattering\" + fname).s;
    s = s(1:numel(f), :);
    s = s(:, 1:end/4);
    fig = figure('Visible','off');
    hold on
    
    D = data(data.File == fname, :);
    t = (0:size(s, 2)-1)/fs;
    Tools.plotTF(s, f, true, true, t);
    for j = 1:size(D, 1)
        detection = D(j, :);
        tbox = ([detection.StartIndex detection.EndIndex]-1)/fs;
        tbox = [tbox fliplr(tbox)];
        fbox = [f(5) f(5) f(end-5) f(end-5)];
        fill(tbox, fbox, 'r-', FaceAlpha=0.2);
        text(mean(tbox), ftext(1), detection.Annotation, Interpreter="none", HorizontalAlignment="center", Color='r');
        for k = 1:numel(classes)
            class = classes(k);
            col = sprintf("Predict_%s", class);
            text(mean(tbox), ftext(k+1), detection.(col), Interpreter="none", HorizontalAlignment="center");
        end
    end    
    xlim([min(t), max(t)]);
    ylim([min(f) max(f)]);
    xticks(0:5:max(t));
    hold off
    fig.Position = [0, 0, 5000000, 500];
    saveas(fig, "detections\" + replace(fname, ".mat", ".jpg"))
    close(fig);
end



