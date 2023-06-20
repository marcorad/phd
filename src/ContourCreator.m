classdef ContourCreator
    properties
        dl DataLoader2
        foldername
        path
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
    end

    methods
        function obj = ContourCreator(foldername)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path + "\\spectrograms", "mat", "parallel", true);
        end



        function create(obj)
            sdir = obj.path + "\\contours\\";
            mkdir(sdir);
            fb = load(obj.path + "\\spectrograms\\filterbank.mat").fb;
            obj.dl.startWaitbar();
            spmd
%                             for spmdIndex = 5
                featureTable = {};
                features = {};
                while ~obj.dl.isComplete(spmdIndex)
%                                     for i = 1:2
                    
                    [s, info, ~] = obj.dl.next(spmdIndex);
                    s = s.^2;
                    
                    c = Contours(fb, s, info);
                    c.fitGMM(true);
                    if c.gmm.converged
                        c.extractContours();
                        T = c.getFeatureTable();
                        if ~isempty(T)
                            featureTable{end + 1} = T;
                            features{end + 1} = c.features;
                        end
                    end
                end
            end
            featureTable = [featureTable{:}];
            features = [features{:}];
            featureTable = vertcat(featureTable{:});
            save(sdir + "ContourTable.mat", "featureTable", '-mat');
            save(sdir + "Features.mat", "features", '-mat');
        end

    end

    methods(Static)
        function parsave(fname, segmm)
            save(fname, 'segmm');
        end
    end
end