function [Xtrain, ytrain, Xtest, ytest] = split(Xn, y, pnoise, psig)
ynoiseidx = find(y == "Noise");
ysigidx = find(y ~= "Noise");
idxnoise = splitlabels(y(ynoiseidx), pnoise, "randomized");
idxsignal = splitlabels(y(ysigidx), psig, "randomized");
idxs{1} = [ynoiseidx(idxnoise{1}); ysigidx(idxsignal{1})];
idxs{2} = [ynoiseidx(idxnoise{2}); ysigidx(idxsignal{2})];
Xtrain = Xn(idxs{1}, :);
ytrain = y(idxs{1}, :);
Xtest = Xn(idxs{2}, :);
ytest = y(idxs{2}, :);
end