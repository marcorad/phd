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
        fb
    end

    methods
        function this = DataLoader2(path, dtype, idxmode, isformatted)
            this.path = path;
            this.dtype = dtype;
            this.isformatted = isformatted;
            this.filelist = this.listFiles();
            this.resetIdx(idxmode);            
        end

        function resetIdx(this, mode)
            if mode == "single"
                this.Idx = ThreadIdx(numel(this.filelist), 1);
            elseif mode == "parallel"
                this.Idx = ThreadIdx(numel(this.filelist), DataLoader2.Nthreads);
            end
        end

        function [x, info, fs] = load(this, idx)
            info = this.filelist(idx);
            
            if this.dtype == "mat"
                x = load(info.path).s;
                if class(this.fb) == "SFB"
                    fs = this.fb.getSSamplingFreq();
                elseif class(this.fb) == "Scattering"
                    fs = this.fb.filterBanks(1).getSSamplingFreq();
                end
            elseif this.dtype == "wav"
                [x, fs] = audioread(info.path);
            end
        end

        function startWaitbar(this)
            wb = waitbar(0);
            parforWaitbar(wb, numel(this.filelist));
            this.wtq = parallel.pool.DataQueue;
            afterEach(this.wtq,@parforWaitbar);
        end

        function [x, info, fs] = next(this, tid)
            if nargin < 2
                tid = 1;
            end
            idx = this.Idx.next(tid);
            [x, info, fs] = this.load(idx);
            if ~isempty(this.wtq)
                send(this.wtq, []);
            end
        end

        function [x, info, fs] = loadFID(this, fid)
            idx = find([this.auxlist.fid] == fid);
            [x, info, fs] = this.load(idx);
        end

        function c = isComplete(this, tid)
            if nargin < 2
                tid = 1;
            end
            c = this.Idx.complete(tid);
        end



    end

    methods(Access=private)
        function filelist = listFiles(this)
            files = dir(this.path);
            files = files(~[files.isdir]);
            files = convertCharsToStrings({files.name});
            idx = arrayfun(@(s) s.contains(sprintf(".%s", this.dtype)), files);
            files = files(idx);
            if this.isformatted && this.dtype == "mat"
                files = files(files ~= "filterbank.mat");
                this.fb = load(this.path + "\\filterbank.mat").fb;
            end
            for i = 1:numel(files)
                s.name = string(files(i));
                s.path = sprintf("%s\\%s", this.path, s.name);
                s.fid = i;
                if this.isformatted
                    s.time = this.filename2datetime(s.name);
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