classdef Detector < handle

    properties
        dlscalogram DataLoader2
        dlscat DataLoader2
        foldername
        path
        noiseEstimate
        annotationTable
        scatfs
        scatfb Scattering
        scat1size
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\"; 
        Tmf = 0.6;
        Tmfscat = 10;
        ScatP = 50;
    end

    methods

        function sw = ScatteringNoiseEstimate(obj, s) 
            M = floor(obj.Tmfscat * obj.scatfs / 2)*2 + 1;
            s = smoothdata(s, 2, "movmedian", M);
            sw = prctile(s, obj.ScatP, 2);
        end

        function obj = Detector(foldername)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
%             obj.dlscalogram = DataLoader2(obj.path + "\\spectrograms", "mat", "parallel", true);
            obj.noiseEstimate = load(obj.path + "\\noise\\NoiseEstimate.mat").noiseEstimates;
            obj.annotationTable = load(obj.path + "\\annotations\\annotations.mat").annotationTable;
%             obj.annotationTable = obj.annotationTable(obj.annotationTable.AnnotationSignificance <= annsig, :);
            obj.dlscat = DataLoader2(obj.path + "\\scattering", "mat", "parallel", true);
            obj.scatfb = load(obj.path + "\\scattering\\filterbank.mat").fb;
            obj.scatfs = obj.scatfb.filterBanks(1).getSSamplingFreq();
            obj.scat1size = numel(obj.scatfb.filterBanks(1).lambdas);
        end

        function detections(obj)
            sdir = obj.path + "\\detectors\\";
            mkdir(sdir);
            obj.dlscat.startWaitbar();
            for spmdIndex = 1
                D = {};            
                while ~obj.dlscat.isComplete(spmdIndex)
                    [s, info, fs] = obj.dlscat.next(spmdIndex);
                    s = s.^2;
                    
                    fid = info.fid;
                    noise = obj.ScatteringNoiseEstimate(s);
                    sw = s./noise;
                    
                    s1 = s(1:obj.scat1size, :);
                    s1w = sw(1:obj.scat1size, :);

                    M = floor(Detector.Tmf * obj.scatfs)*2 + 1;
                    detectors.segmms1 = SEGMM(M);
                    detectors.segmms1white = SEGMM(M);
                    detectors.segmmscat = SEGMM(M);
                    detectors.segmmscatwhite = SEGMM(M);
                    detectors.bled = BLED();
                    detectors.bledwhite = BLED();

                    detectors.segmms1.detect(s1);
                    detectors.segmms1white.detect(s1w);
                    detectors.segmmscat.detect(s);
                    detectors.segmmscatwhite.detect(sw);
                    detectors.bled.detect(s1./obj.scatfb.filterBanks(1).lambdas');
                    detectors.bledwhite.detect(s1w);
                    detectors.info = info;
                    
                    anns = obj.annotationTable(obj.annotationTable.FileID == fid, :);
                    detectors.annotations = anns;

                    truelabels = zeros(1, size(s1, 2));
                    for i = 1:size(anns, 1)
                        sidx = anns.ScatteringStartIndex(i);
                        eidx = min(anns.ScatteringEndIndex(i), size(s, 2));
                        truelabels(sidx:eidx) = 1;
                    end
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