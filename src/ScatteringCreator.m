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
        R = 1;
    end

    methods
        function this = ScatteringCreator(foldername, Q, fl, fh, T, oversample)
            this.foldername = foldername;
            this.path = ScatteringCreator.datapath + string(foldername);
            this.dl = DataLoader2(this.path, "wav", "single", true);
            this.fb = Scattering(Q, T, 250, AudioDataConverter.L, fl, fh, oversample);
            sdir = this.path + "\\scattering\\";
            mkdir(sdir); 
        end

        function create(this)          
            this.dl.startWaitbar();
            sdir = this.path + "\\scattering\\";
            while ~this.dl.isComplete()
                [x, info, ~] = this.dl.next();
                x = resample(x, 1, this.R); %resample to 250Hz for speed
                s = this.fb.scattering(x);
                name = string(info.time) + ".mat";
                save(sdir + name, "s", '-mat');
            end   
            fb = this.fb;
            save(sdir + "filterbank.mat", "fb", '-mat');            
        end
    end
end