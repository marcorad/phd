classdef Detector < handle

    properties
        dl DataLoader2
        foldername
        path
        noiseEstimate
        annotationTable
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\"; 
        Tmf = 0.6;
        
    end

    methods
        function obj = Detector(foldername, annsig)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path + "\\spectrograms", "mat", "parallel", true);
            obj.noiseEstimate = load(obj.path + "\\noise\\NoiseEstimate.mat").noiseEstimates;
            obj.annotationTable = load(obj.path + "\\annotations\\annotations.mat").annotationTable;
            obj.annotationTable = obj.annotationTable(obj.annotationTable.AnnotationSignificance <= annsig, :);
        end

        function detections(obj)
            sdir = obj.path + "\\detectors\\";
            mkdir(sdir);
            obj.dl.startWaitbar();
            spmd %for spmdIndex = 1
                D = {};            
                while ~obj.dl.isComplete(spmdIndex)
                    [s, info, fs] = obj.dl.next(spmdIndex);
                    s = s.^2;
                    fid = info.fid;
                    noise = obj.noiseEstimate(:, fid);
                    sw = s./noise;
                    swg = imgaussfilt(sw, [0.75, 3]);
                    M = floor(Detector.Tmf * fs)*2 + 1;
                    detectors.segmm = SEGMM(M);
                    detectors.segmmwhite = SEGMM(M);
                    detectors.segmmwhitegauss = SEGMM(M);
                    detectors.bled = BLED();
                    detectors.bledwhite = BLED();
                    detectors.segmm.detect(s);
                    detectors.bled.detect(s);
                    detectors.segmmwhite.detect(sw);
                    detectors.segmmwhitegauss.detect(swg);
                    detectors.bledwhite.detect(sw);
                    detectors.info = info;
                    
                    anns = obj.annotationTable(obj.annotationTable.FileID == fid, :);
                    truelabels = zeros(1, size(s, 2));
                    for i = 1:size(anns, 1)
                        sidx = anns.SpectrogramStartIndex(i);
                        eidx = anns.SpectrogramEndIndex(i);
                        truelabels(sidx:eidx) = 1;
                    end
                    detectors.annotations = anns;
                    detectors.truelabels = truelabels;

                    D{end + 1} = detectors;
                end
        
            end
            D = [D{:}];
            D = [D{:}];
            save(sdir + "\\detectors.mat", "D", '-mat');
        end
    end

end