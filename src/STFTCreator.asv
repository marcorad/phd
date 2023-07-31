classdef STFTCreator
    properties
        dl DataLoader2
        foldername
        path
    end

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
    end

    methods
        function obj = STFTCreator(foldername)
            obj.foldername = foldername;
            obj.path = STFTCreator.datapath + string(foldername);
            obj.dl = DataLoader2(obj.path, "wav", "parallel", true);
        end

        function saveSTFT(obj, name, s)
            sdir = obj.path + "\\stft\\";
            save(sdir + name, "s", '-mat');
        end

        function create(obj, Nfft, Noverlap, fs)
            sdir = obj.path + "\\stft\\";
            mkdir(sdir);            
            obj.dl.startWaitbar();
            spmd
                while ~obj.dl.isComplete(spmdIndex)
                    [x, info, ~] = obj.dl.next(spmdIndex);
                    w = hann(Nfft);
                    s = stft(x, fs, "Window",w, "OverlapLength",Noverlap, "FFTLength",Nfft, "OutputTimeDimension","acrosscolumns", "FrequencyRange","onesided");                
                    name = string(info.time) + ".mat";
                    obj.saveSTFT(name, s);
                end   
            end
        end
    end
end