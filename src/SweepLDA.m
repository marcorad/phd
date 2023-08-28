
function sweepresults = SweepLDA()

%load the data
folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
data = load(folder + "\features\features.mat").data;

classes_to_sweep = ["Bm_Ant_A", "Bm_Ant_B", "Bm_Ant_Z", "Bm_D", "Bp_20Plus"];

sweepresults = {};

for i = 1:numel(classes_to_sweep)

    class = classes_to_sweep(i)

    %prepare the data
    X = data{:, "Features"};
    X = double(X);
    y = data{:, "Annotation"};

    y = replace(y, "-", "_");

    %remove irrelevant classes
    idx = ~contains(y, "Multiple");
    idx = idx & all(~isnan(X), 2);
    if contains(class, "Bm_Ant")
        %ignore other Bm_Ant classes in the evaluation
        idx = idx & ~(contains(y, "Bm_Ant") & y ~= class);
    end

    y = y(idx);

    y(~contains(y, class)) = "Noise";
    X = X(idx, [1:end-1, end]);
    
    Xn = ((X-mean(X, 1))./std(X, 1));

    %sweep hyperparams
    noiseps = linspace(0.05, 0.9, 5);
    gammas = linspace(0.1, 1, 5);
    C = numel(unique(y));
    cm = zeros(C, C, numel(noiseps), numel(gammas));
    varcm = cm;
    
    score = zeros(numel(noiseps), numel(gammas));
    N = 20;
    parfor j = 1:N     
        fprintf("%d", j);
        [cmt, scoret] = sweep(Xn, y, noiseps, gammas, 2);
        cm = cm + cmt/N;
        varcm = varcm + (cmt./sum(cmt, 2)).^2/N;
        score = score + scoret/N;
    end

    fprintf("\n");

    stdcm = sqrt(varcm - (cm./sum(cm, 2)).^2);

    %plot
%     figure
%     contour(gammas, noiseps , score, 0.1:0.02:100, "ShowText","on")
%     xlabel("\gamma")
%     ylabel("\alpha")
%     zlabel("Average Accuracy")
%     title(class, Interpreter="none");
    
    [mscore, midx] = max(score, [], "all");
    [aidx, gammaidx] = ind2sub(size(score), midx);
    fprintf("Max score of %.2f with parameters a=%.2f and gamma=%.2f\n", mscore * 100, noiseps(aidx), gammas(gammaidx))

    
    
    cmbest = cm(:, :, aidx, gammaidx);
    cmnorm = cmbest ./ sum(cmbest, 2);
    

    fprintf("Resulting accuracies of %.2f (noise) and %.2f (signal)\n", cmnorm(1,1), cmnorm(2,2))
%     figure
%     confusionchart(floor(cmbest), sort(unique(y), 1, "descend"), "RowSummary", "row-normalized")
    
    result.Class = class;
    result.SignalAccuracy = cmnorm(2,2) * 100;
    result.NoiseAccuracy = cmnorm(1,1) * 100;
    result.Alpha = noiseps(aidx);
    result.Gamma = gammas(gammaidx);
    result.StdSignalAccuracy = stdcm(2,2) * 100;
    result.StdNoiseAccuracy = stdcm(1,1) * 100;

    sweepresults{end + 1} = result;

end
    sweepresults = vertcat(sweepresults{:});
    sweepresults = struct2table(sweepresults)

end



function [cm, score] = sweep(Xn, y, noiseps, gammas, lambda)

C = numel(unique(y));
cm = zeros(C, C, numel(noiseps), numel(gammas));
sigvsnoise = zeros(2, 2, numel(noiseps), numel(gammas));
score = zeros(numel(noiseps), numel(gammas));
for i = 1:numel(noiseps)
    noisep = noiseps(i);
    for j = 1:numel(gammas)
        gamma = gammas(j);
        sigp = (1- noisep);
        prior = struct("Bm_Ant_A",sigp,"Bm_Ant_B",sigp,"Bm_Ant_Z",sigp,"Bm_Ant",sigp, "Bm_D", sigp, "Bp_20Plus", sigp, "Noise", noisep);
        covtype = struct("Bm_Ant_A","full","Bm_Ant_B","full","Bm_Ant_Z","full","Bm_Ant","full", "Bm_D", "full", "Bp_20Plus", "full", "Noise", "diag");
        cls = Discriminant(gamma, "linear", 0);        
%         [Xtrain, ytrain, Xtest, ytest] = split(Xn, y, 3000, min(100, floor(0.8*sum(y ~= "Noise"))));
        [Xtrain, ytrain, Xtest, ytest] = split(Xn, y, 0.8, 0.8);
        cls.fit(Xtrain, ytrain, prior, covtype);
        yhat = cls.predict(Xtest);
        [c, order] = confusionmat(ytest, yhat);
        [order, sidx] = sort(order, 1, "descend");
        c = c(sidx, sidx);
        cm(:, :, i, j) = c;

        cmnorm = c./sum(c, 2);
        %         score(i, j) = mean(diag(c./sum(c, 2)));
        score(i, j) = sum(diag(cmnorm).*[lambda; 1])/(1 + lambda);
    end
end



end
