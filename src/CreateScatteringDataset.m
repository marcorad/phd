function CreateScatteringDataset()

folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
dl = DataLoader2(folder + "\scattering", 'mat', 'parallel', true);
annotations = load(folder + "\annotations\annotations.mat").annotationTable;
fb = load(folder + "\scattering\filterbank.mat").fb;
Ns1 = numel(fb.filterBanks(1).lambdas);

dl.startWaitbar();

spmd%for spmdIndex = 1
    data = cell2table(cell(0, 6), 'VariableNames',["File", "Annotation", "Features", "AnnotationPower", "StartIndex", "EndIndex"]);
    counter = 0;
    while ~dl.isComplete(spmdIndex)% && counter < 5
        [s, info, fs] = dl.next(spmdIndex);
        ann = annotations(annotations.File == replace(info.name, ".mat", ".wav"), :);
        s = s.^2;
        s1 = s(1:Ns1, :);
        sw = noiseEstimate(s1, 9, floor(120/2*fs)*2 + 1);
        s1w = s1./sw;
        segmm = SEGMM(floor(1.5*fs/2)*2 + 1);
        segmm.detect(s1w);
        
        if ~segmm.converged || segmm.Hmax - segmm.Hmin <= snr2dh(db2mag(-10), Ns1)
            continue;
        end

        rle = Tools.rle(Tools.threshhyst(segmm.probs, 0.6, 0.3));
        extend_n = floor(fs*5); 
        %extend the detections
        rle.Start = max(rle.Start - extend_n, 1);
        rle.End = min(rle.End + extend_n, size(s, 2));
        for i = 1:size(rle, 1)
            r = rle(i, :);
            dur = ann.ScatteringEndIndex - ann.ScatteringStartIndex;
            anncomp = dur * 0.05; %compensate for inaccurate annotation localisation
            ovr = r.Start <= ann.ScatteringEndIndex - anncomp & ann.ScatteringStartIndex + anncomp <= r.End;
            ss = r.Start;
            se = r.End;
%             coeffs = mean(s(:, ss:se),  2);
            coeffs = max(s(:, ss:se), [],  2);
            coeffs = cosineLogScattering(coeffs, fb.pathstart);
            row.Features = [coeffs, r.End-r.Start];
            row.Annotation = "Noise";
            row.AnnotationPower = 0;
            row.StartIndex = r.Start;
            row.EndIndex = r.End;
            row.File = info.name;
            if sum(ovr) == 1
                row.Annotation = ann.Annotation(ovr);
                fc = fb.filterBanks(1).fc;
                currann = ann(ovr, :);
                fidx = fc >= currann.StartFrequency & fc <= currann.EndFrequency;                
                tidx = currann.ScatteringStartIndex:min(currann.ScatteringEndIndex, size(s1w, 2));
                row.AnnotationPower = mean(s1w(fidx,tidx), "all");
            elseif sum(ovr) > 1
                row.Annotation = "Multiple";
            end
           data = [data; struct2table(row)];
        end

        counter = counter + 1;
        
    end
    
end

data = data(:);
data = vertcat(data{:});

groupcounts(data, "Annotation")

save(folder + "\features\features.mat", "data", '-mat');

end

function cls = cosineLogScattering(coeffs, pathstart)
    coeffs = log(coeffs);
    pathstart = [pathstart, numel(coeffs)+1];
    cls = zeros(1, numel(coeffs));
    for i = 1:numel(pathstart)-1
        s = pathstart(i);
        e = pathstart(i+1) - 1;
        cls(s:e) = dct(coeffs(s:e))'; %compute along each path
    end
end