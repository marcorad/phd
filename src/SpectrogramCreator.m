classdef SpectrogramCreator
    properties
        dl DataLoader2
        foldername
        path
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
    end

    methods
        function obj = SpectrogramCreator(foldername)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path, "wav", "single", true);
        end

        function create(obj, Q, T, fs, N, fl, fh)
            fb = SFB(Q, T, fs, N, fl, fh);
            sdir = obj.path + "\\spectrograms\\";
            mkdir(sdir);            
            obj.dl.startWaitbar();
            while ~obj.dl.isComplete()
                [x, info, ~] = obj.dl.next();
                s = fb.filterS(x);
                name = string(info.time) + ".mat";
                save(sdir + name, "s", '-mat');
            end   
            save(sdir + "filterbank.mat", "fb", '-mat');
        end
    end
end