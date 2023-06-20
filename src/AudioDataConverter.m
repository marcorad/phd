classdef AudioDataConverter < handle
    properties(Constant)
        L = 2^18; %number of samples
        dtoutputformat = "yyyy_MM_dd_HH_mm_ss_SSS"        
    end
    properties
        inputpath %the path to the wav files
        outputpath %the path at which to output the files
        dtinputformat %the datetime format of the input path naming scheme
        fs %the target sampling frequency
        
    end

    methods
        function obj = AudioDataConverter(path, outputpath, dtinputformat, fs)
            obj.inputpath = path;    
            obj.outputpath = outputpath;
            obj.dtinputformat = dtinputformat;
            obj.fs = fs;
        end

        function convert(obj)
            mkdir(obj.outputpath);
            dl = DataLoader2(obj.inputpath, "wav", 'parallel', false);
            dl.startWaitbar();
            spmd %parfor spmdIndex = 1:6
                while ~dl.isComplete(spmdIndex)
                    [x, info, fsOrig] = dl.next(spmdIndex);                                    
                    [p,q] = rat(obj.fs/fsOrig);
                    x = resample(x,p,q);
                    D = AudioDataConverter.partitionAudio(x, AudioDataConverter.L);
                    fname = info.name;
                    tstr = fname.extractBefore(find(fname{:} == '.'));
                    tstr = tstr.extractBefore(strlength(obj.dtinputformat)+1);
                    t = datetime(tstr, "InputFormat",obj.dtinputformat, "Format", AudioDataConverter.dtoutputformat);
                    dt = seconds(AudioDataConverter.L/obj.fs);
                    for i = 1:numel(D)
                        outt = t + (i-1)*dt;
                        outname = sprintf("%s.wav", string(outt));
                        audiowrite(sprintf("%s\\%s", obj.outputpath, outname), D{i}, obj.fs);
                    end
                end
            end
        end        
    end

    methods(Static)
        function D = partitionAudio(x, L)
            sidx = 1;
            N = numel(x);
            i = 1;
            while true
                if sidx > N
                    break;
                end
                eidx = min(sidx + L - 1, N);
                D{i} = x(sidx:eidx);
                sidx = sidx + L;
                i = i+1;
            end
        end      
        
    end
end