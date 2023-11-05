function CreateScatteringDataset()

folder = "D:\Whale Data\Raw Audio Data\CaseyIslands2017";
dl = DataLoader2(folder + "\scattering", 'mat', 'parallel', true);
annotations = load(folder + "\annotations\annotations.mat").annotationTable;
fb = load(folder + "\scattering\filterbank.mat").fb;
Ns1 = numel(fb.filterBanks(1).lambdas);

annotations.Annotation(contains(annotations.Annotation, "Bp")) = "Bp";

dl.startWaitbar();

spmd %for spmdIndex = 1
    data = cell2table(cell(0, 6), 'VariableNames',["File", "Annotation", "Features", "AnnotationPower", "StartIndex", "EndIndex"]);
    counter = 0;
    while ~dl.isComplete(spmdIndex)% && counter < 5
        [s, info, fs] = dl.next(spmdIndex);
        ann = annotations(annotations.File == replace(info.name, ".mat", ".wav"), :);
        
        s1 = s(1:Ns1, :);
        sw = noiseEstimate(s1, fs);
        s1w = s1./sw;
        segmm = SEGMM(floor(0.75*fs)*2 + 1);
        s1w = s1w.^2;
        segmm.detect(s1w);
        
        if ~segmm.converged
            continue;
        end
        dets = segmm.probs > 0.3;
        %extend the detections
        extend_n = floor(fs*3)*2 + 1; 
        dets = movmax(dets, extend_n);
        rle = Tools.rle(dets);       
%         rle.Start = max(rle.Start - extend_n, 1);
%         rle.End = min(rle.End + extend_n, size(s, 2));
        for i = 1:size(rle, 1)
            r = rle(i, :);
            dur = ann.ScatteringEndIndex - ann.ScatteringStartIndex;
            anncomp = dur * 0.05; %compensate for inaccurate annotation localisation
            ovr = r.Start <= ann.ScatteringEndIndex - anncomp & ann.ScatteringStartIndex + anncomp <= r.End;
            ss = r.Start;
            se = r.End;
%             coeffs = mean(s(:, ss:se),  2);
            lambda1 = fb.filterBanks(1).lambdas';
            pt = sum(s(1:numel(lambda1), ss:se).^2./lambda1, 1);
            p = mean(pt, 2); %RMS
            coeffs = max(s(:, ss:se), [],  2)/p; %normalise by RMS
            coeffs = cosineLogScattering(coeffs, fb.pathstart);
            row.Features = [coeffs, r.End-r.Start];
            row.Annotation = "Noise";
            row.AnnotationPower = 0;
            row.StartIndex = r.Start;
            row.EndIndex = r.End;
            row.File = info.name;
            allsame = false;
            first_idx = find(ovr, 1);
            nann = sum(ovr);
            if nann > 1
                allsame = all(ann.Annotation(ovr) == ann.Annotation(first_idx));
            end
            if nann == 1 || allsame
                row.Annotation = ann.Annotation(first_idx);
                fc = fb.filterBanks(1).fc;
                currann = ann(first_idx, :);
                fidx = fc >= currann.StartFrequency & fc <= currann.EndFrequency;                
                tidx = currann.ScatteringStartIndex:min(currann.ScatteringEndIndex, size(s1w, 2));
                row.AnnotationPower = mean(s1w(fidx,tidx), "all");
            elseif nann > 1
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