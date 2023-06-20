classdef SEGMMCreator
    properties
        dl DataLoader2
        foldername
        path
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
        Tmf = 0.1;
    end

    methods
        function obj = SEGMMCreator(foldername)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path + "\\spectrograms", "mat", "parallel", true);
        end



        function create(obj)
            sdir = obj.path + "\\segmm\\";
            sdirwhite = sdir + "white\\";
            sdirnotwhite = sdir + "notwhite\\";
            mkdir(sdir);
            mkdir(sdirwhite);
            mkdir(sdirnotwhite);
            obj.dl.startWaitbar();
            spmd
%                             for spmdIndex = 1
                segmmTableWhite = {};
                segmmTableNotWhite = {};
                noiseEstimates = {};
                t = {};
                while ~obj.dl.isComplete(spmdIndex)
%                 for i = 1:2
                    [s, info, ~] = obj.dl.next(spmdIndex);
                    s = s.^2;
                    smed = Contours.estimateNoise(s);

                    t{end+1} = info.time;
                    noiseEstimates{end + 1} = smed;


                    M = floor(SEGMMCreator.Tmf * fbfs/2)*2 + 1;

                    %whitened spectrogram detection
                    segmmwhite = SEGMM(M);
                    segmmwhite.detect(s./smed);
                    segmmTableWhite{end + 1} = segmmwhite.getStatistics();
                    SEGMMCreator.parsave(sdirwhite + info.name, segmmwhite);

                    %non-whitened spectrogram detection
                    segmmnotwhite = SEGMM(M);
                    segmmnotwhite.detect(s);
                    segmmTableNotWhite{end + 1} = segmmnotwhite.getStatistics();
                    SEGMMCreator.parsave(sdirnotwhite + info.name, segmmnotwhite);
                end                
            end
            segmmTableWhite = struct2table(cell2mat([segmmTableWhite{:}]));
            segmmTableNotWhite = struct2table(cell2mat([segmmTableNotWhite{:}]));
            save(sdir + "SEGMM.mat", 'segmmTableWhite', 'segmmTableNotWhite');
            noiseEstimates = [noiseEstimates{:}];
            noiseEstimates = [noiseEstimates{:}];
            t = [t{:}]; %for some reason, we must do this twice
            t = [t{:}];
            [t, sidx] = sort(t);
            noiseEstimates = noiseEstimates(:, sidx);
            save(sdir + "NoiseEstimate.mat", 't', 'noiseEstimates');

        end
    end

    methods(Static)
        function parsave(fname, segmm)
            save(fname, 'segmm');
        end
    end
end