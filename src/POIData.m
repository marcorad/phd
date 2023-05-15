classdef POIData < handle
    properties
        featsAug = load("store\Features1.mat", "feats");
        featsOct = load("store\Features2.mat", "feats");
        pathAug = load("store\Features1.mat", "path");
        pathOct = load("store\Features2.mat", "path");
    end

    methods

        function getData(obj, dataset, freq, cluster, idx)
            if dataset == "August"
                feats = obj.featsAug;
                path = obj.pathAug;
            elseif dataset == "October"
                feats = obj.featsOct;
                path = obj.pathOct;
            end



        end
        
    end

end