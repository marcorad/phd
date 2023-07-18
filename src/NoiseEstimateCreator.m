classdef NoiseEstimateCreator < handle
properties
        dl DataLoader2
        foldername
        path
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\"; 
    end

    methods
        function obj = NoiseEstimateCreator(foldername)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path + "\\spectrograms", "mat", "parallel", true);
        end

        function create(obj)
            sdir = obj.path + "\\noise\\";
            mkdir(sdir);
            obj.dl.startWaitbar();
            spmd
                noiseEstimates = {};
                noiseVarianceEstimates = {};
                t = {};
                while ~obj.dl.isComplete(spmdIndex)
%                 for i = 1:2
                    [s, info, ~] = obj.dl.next(spmdIndex);
                    s = s.^2;
                    smed = Contours.estimateNoise(s);
                    svar = Contours.estimateNoiseVar(s);

                    t{end+1} = info.time;
                    noiseEstimates{end + 1} = smed;
                    noiseVarianceEstimates{end + 1} = svar;
                end                
            end
            noiseEstimates = [noiseEstimates{:}];
            noiseEstimates = [noiseEstimates{:}];
            noiseVarianceEstimates = [noiseVarianceEstimates{:}];
            noiseVarianceEstimates = [noiseVarianceEstimates{:}];
            t = [t{:}]; %for some reason, we must do this twice
            t = [t{:}];
            [t, sidx] = sort(t);
            noiseEstimates = noiseEstimates(:, sidx);
            noiseVarianceEstimates = noiseVarianceEstimates(:, sidx);
            save(sdir + "NoiseEstimate.mat", 't', 'noiseEstimates');
            save(sdir + "NoiseVarianceEstimate.mat", 't', 'noiseVarianceEstimates');
        end
    end

end