classdef ScatteringCreator
    properties
        dl DataLoader2
        foldername
        path
        fb Scattering
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
        Tmul = 6;
    end

    methods
        function obj = ScatteringCreator(foldername, Q, fl, fh)
            obj.foldername = foldername;
            obj.path = ScatteringCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path, "wav", "single", true);
            sfb = load(obj.path + "\\spectrograms\\filterbank.mat").fb;
            obj.fb = Scattering(Q, sfb.T*ScatteringCreator.Tmul, sfb.fs, sfb.N, fl, fh);
            sdir = obj.path + "\\scattering\\";
            mkdir(sdir);  
            fb = obj.fb;
            save(sdir + "filterbank.mat", "fb", '-mat');
        end

        function create(obj)          
            obj.dl.startWaitbar();
            sdir = obj.path + "\\scattering\\";
            while ~obj.dl.isComplete()
                [x, info, ~] = obj.dl.next();
                s = obj.fb.scattering(x);
                name = string(info.time) + ".mat";
                save(sdir + name, "s", '-mat');
            end   
            
        end
    end
end