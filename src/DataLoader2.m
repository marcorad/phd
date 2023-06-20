classdef DataLoader2 < handle
    properties(Constant)
        Nthreads = 6;
    end

    properties
        path
        dtype %type of data to load "wav"|"mat"
        filelist
        Idx ThreadIdx
        wtq
        isformatted %whether the data is formatted to the dataset convention in AudioDataConverter
        fb SFB
    end

    methods
        function obj = DataLoader2(path, dtype, idxmode, isformatted)
            obj.path = path;
            obj.dtype = dtype;
            obj.isformatted = isformatted;
            obj.filelist = obj.listFiles();
            obj.resetIdx(idxmode);            
        end

        function resetIdx(obj, mode)
            if mode == "single"
                obj.Idx = ThreadIdx(numel(obj.filelist), 1);
            elseif mode == "parallel"
                obj.Idx = ThreadIdx(numel(obj.filelist), DataLoader2.Nthreads);
            end
        end

        function [x, info, fs] = load(obj, idx)
            info = obj.filelist(idx);
            if obj.dtype == "mat"
                x = load(info.path).s;
                fs = obj.fb.getSSamplingFreq();
            elseif obj.dtype == "wav"
                [x, fs] = audioread(info.path);
            end
        end

        function startWaitbar(obj)
            wb = waitbar(0);
            parforWaitbar(wb, numel(obj.filelist));
            obj.wtq = parallel.pool.DataQueue;
            afterEach(obj.wtq,@parforWaitbar);
        end

        function [x, info, fs] = next(obj, tid)
            if nargin < 2
                tid = 1;
            end
            idx = obj.Idx.next(tid);
            [x, info, fs] = obj.load(idx);
            if ~isempty(obj.wtq)
                send(obj.wtq, []);
            end
        end

        function [x, info, fs] = loadFID(obj, fid)
            idx = find([obj.auxlist.fid] == fid);
            [x, info, fs] = obj.load(idx);
        end

        function c = isComplete(obj, tid)
            if nargin < 2
                tid = 1;
            end
            c = obj.Idx.complete(tid);
        end



    end

    methods(Access=private)
        function filelist = listFiles(obj)
            files = dir(obj.path);
            files = files(~[files.isdir]);
            files = convertCharsToStrings({files.name});
            idx = arrayfun(@(s) s.contains(sprintf(".%s", obj.dtype)), files);
            files = files(idx);
            if obj.isformatted && obj.dtype == "mat"
                files = files(files ~= "filterbank.mat");
                obj.fb = load(obj.path + "\\filterbank.mat").fb;
            end
            for i = 1:numel(files)
                s.name = string(files(i));
                s.path = sprintf("%s\\%s", obj.path, s.name);
                s.fid = i;
                if obj.isformatted
                    s.time = obj.filename2datetime(s.name);
                end
                filelist(i) = s;
            end
            [~, sidx] = sort([filelist.name]);
            filelist = filelist(sidx);
        end
    end

    methods(Static)
        function t = filename2datetime(fname)
            tstr = fname.extractBefore(find(fname{:} == '.'));
            t = datetime(tstr, "InputFormat",AudioDataConverter.dtoutputformat, "Format", AudioDataConverter.dtoutputformat);
        end

        function folders = listFolders(path)
            folders = dir(path);
            folders = folders([folders.isdir]);
            folders = convertCharsToStrings({folders.name});
            folders = folders(3:end);
        end

        function [audiopaths, spectrogrampaths] = availableDatasets()
            audiopath = "D:\\Whale Data\\Raw Audio Data\\";
            spectrogrampath = "D:\\Whale Data\\Raw Audio Data\\spectrograms\\";
            audiopaths = DataLoader2.listFolders(audiopath);
            spectrogrampaths = DataLoader2.listFolders(spectrogrampath);
        end
    end
end