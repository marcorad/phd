classdef Detector < handle

    properties
        dlscalogram DataLoader2
        dlscat DataLoader2
        foldername
        path
        annotationTable
        scatfs
        scatfb Scattering
        scat1size        
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\"; 
        Tmf = 1.5;
        Tmfscat = 10;
        ScatP = 50;
    end

    methods

        function snoise = ScatteringNoiseEstimate(this, s1) 
            s1smooth = smoothdata(s1, 1, "movmedian", 9);
            s1smooth = smoothdata(s1smooth, 2, "movmedian", floor(this.scatfs*5));
            snoise = smoothdata(s1smooth, 2, "movmean", floor(this.scatfs*60));
        end

        function this = Detector(foldername)
            this.foldername = foldername;
            this.path = SpectrogramCreator.datapath + string(foldername);
%             this.dlscalogram = DataLoader2(this.path + "\\spectrograms", "mat", "parallel", true);
            this.annotationTable = load(this.path + "\\annotations\\annotations.mat").annotationTable;
%             this.annotationTable = this.annotationTable(this.annotationTable.AnnotationSignificance <= annsig, :);
            this.dlscat = DataLoader2(this.path + "\\scattering", "mat", "parallel", true);
            this.scatfb = load(this.path + "\\scattering\\filterbank.mat").fb;
            this.scatfs = this.scatfb.filterBanks(1).getSSamplingFreq();
            this.scat1size = numel(this.scatfb.filterBanks(1).lambdas);
        end

        function detector = detect(this, s)
            s = s.^2;
            noise = this.ScatteringNoiseEstimate(s, this.scat1size, this.scatfb.pathstart);
            sw = s./noise;                    
            s1 = s(1:this.scat1size, :);
            s1w = sw(1:this.scat1size, :);
            detector.segmms1 = SEGMM(M);
            detector.segmms1white = SEGMM(M);
            detector.kmeans = SEGMM(M);
            detector.bled = BLED();
            detector.bledwhite = BLED();
            

            detector.kmeans.detect(s1, true);
            detector.segmms1.detect(s1);
            detector.segmms1white.detect(s1w);
            detector.bled.detect(s1./this.scatfb.filterBanks(1).lambdas');
            detector.bledwhite.detect(s1w);
            
        end

        function detections(this)
            sdir = this.path + "\\detectors\\";
            mkdir(sdir);
            this.dlscat.startWaitbar();
            spmd%for spmdIndex = 1
                D = {};            
                while ~this.dlscat.isComplete(spmdIndex)
                    [s, info, fs] = this.dlscat.next(spmdIndex);
                    ssq = s.^2;
                    s1 = s(1:this.scat1size, :); 
                    s1sq = ssq(1:this.scat1size, :);                    
                    
                    fname = replace(info.name, ".mat", ".wav");
                    noise = this.ScatteringNoiseEstimate(s1);
                    s1w = s1./noise;
                    s1wsq = s1w.^2;

                    M = floor(Detector.Tmf * this.scatfs)*2 + 1;
                    detectors.segmms1 = SEGMM(M);
                    detectors.segmms1white = SEGMM(M);
                    detectors.kmeans = SEGMM(M);
                    detectors.bled = BLED();
                    detectors.bledwhite = BLED();

                    detectors.kmeans.detect(s1sq, true);
                    detectors.segmms1.detect(s1sq);
                    detectors.segmms1white.detect(s1w);
                    detectors.bled.detect(s1sq./this.scatfb.filterBanks(1).lambdas');
                    detectors.bledwhite.detect(s1wsq);
                    detectors.bledwhite.probs = smoothdata(detectors.bledwhite.probs, 2, "movmedian", M);
                    detectors.info = info;
                    
                    anns = this.annotationTable(this.annotationTable.File == fname, :);
                    detectors.annotations = anns;

                    truelabels = zeros(1, size(s1sq, 2));
                    for i = 1:size(anns, 1)
                        sidx = anns.ScatteringStartIndex(i);
                        eidx = min(anns.ScatteringEndIndex(i), size(ssq, 2));
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