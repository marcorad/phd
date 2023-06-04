classdef DataLoader < handle
    properties(Constant)
        audiopath = "D:\Whale Data\POIData";
        spectrogrampath = "D:\Whale Data\POIData\spectrograms";
        Nthreads = 6;
    end

    properties
        ds

        audiolist
        spectrogramlist
        auxlist

        audioIdx
        spectroIdx
        auxIdx

        audiowtq
        spectrowtq

        fb

    end

    methods
        function obj = DataLoader(ds, mode)
            if nargin < 2
                mode = "single";
            end
            obj.ds = ds;
            obj.audiolist = obj.listAudio();
            obj.spectrogramlist = obj.listSpectrograms();
            obj.resetIdx(mode);
            load(sprintf("%s\\filterbank.mat", DataLoader.spectrogrampath));
            obj.fb = fb;
        end

        function auxLoader(infolist)

        end

        function resetIdx(obj, mode)
            if mode == "single"
                obj.audioIdx = ThreadIdx(numel(obj.audiolist), 1);
                obj.spectroIdx = ThreadIdx(numel(obj.spectrogramlist), 1);
            elseif mode == "parallel"
                obj.audioIdx = ThreadIdx(numel(obj.audiolist), DataLoader.Nthreads);
                obj.spectroIdx = ThreadIdx(numel(obj.spectrogramlist), DataLoader.Nthreads);
            end
        end

        function [D, info] = loadAudio(obj, idx)
            info = obj.audiolist(idx);
            x = audioread(info.path);
            L = 2^18;
            s = 1;
            e = L;
            for i = 1:16
                if e > numel(x)
                    D{i} = x(s:end);
                    break;
                end
                D{i} = x(s:e);
                s = s + L;
                e = e + L;
            end
        end

        function [s, info] = loadSpectrogram(obj, idx)
            info = obj.spectrogramlist(idx);
            load(info.path);
        end

        function startAudioWaitbar(obj)
            wb = waitbar(0);
            parforWaitbar(wb, numel(obj.audiolist));
            obj.audiowtq = parallel.pool.DataQueue;
            afterEach(obj.audiowtq,@parforWaitbar);
        end

        function startSpectrogramWaitbar(obj)
            wb = waitbar(0);
            parforWaitbar(wb, numel(obj.spectrogramlist));
            obj.spectrowtq = parallel.pool.DataQueue;
            afterEach(obj.spectrowtq,@parforWaitbar);
        end

        function [D, info] = nextAudio(obj, tid)
            if nargin < 2
                tid = 1;
            end
            idx = obj.audioIdx.next(tid);
            [D, info] = obj.loadAudio(idx);
            if ~isempty(obj.audiowtq)
                send(obj.audiowtq, []);
            end
        end

        function [s, info] = nextSpectrogram(obj, tid)
            if nargin < 2
                tid = 1;
            end
            idx = obj.spectroIdx.next(tid);
            [s, info] = obj.loadSpectrogram(idx);
            if ~isempty(obj.spectrowtq)
                send(obj.spectrowtq, []);
            end
        end

        function c = isAudioComplete(obj, tid)
            if nargin < 2
                tid = 1;
            end
            c = obj.audioIdx.complete(tid);
        end

        function c = isSpectrogramComplete(obj, tid)
            if nargin < 2
                tid = 1;
            end
            c = obj.spectroIdx.complete(tid);
        end

    end

    methods(Access=private)
        function L = listAudio(obj)
            p = sprintf("%s\\%d", DataLoader.audiopath, obj.ds);
            files = dir(p);
            files = files(~[files.isdir]);
            dataset = obj.ds;
            parfor i = 1:numel(files)
                name = string(files(i).name);
                path = sprintf("%s\\%s", p, name);
                time = datetime(extractBetween(name, 1, 8+6+4+2), ...
                    "InputFormat","yyyyMMdd_HHmmss_SSSS");
                L(i) = struct("name", name, "path", path, "time", time, "ds", dataset, "fid", i);
            end
            [~, sidx] = sort([L.time]);
            L = L(sidx);
            if ~isempty(L)
                [~, sidx] = sort([L.time]);
                L = L(sidx);
            end
        end

        function L = listSpectrograms(obj)
            p = sprintf("%s\\%d", DataLoader.spectrogrampath, obj.ds);
            files = dir(p);
            segT = seconds(2^18/2000);
            files = files(~[files.isdir]);
            dataset = obj.ds;
            parfor i = 1:numel(files)
                name = string(files(i).name);
                path = sprintf("%s\\%s", p, name);
                time = datetime(extractBetween(name, 1, 8+6+4+2), ...
                    "InputFormat","yyyyMMdd_HHmmss_SSSS");
                seg = regexp( extractAfter(name, 8+6+4+2), '\d+', ...
                    'match', 'once' );
                seg = str2double(seg);
                time = time + (seg-1)*segT;
                L(i) = struct("name", name, "path", path, "time", time, "ds", dataset, "fid", i);
            end
            if exist("L")
                [~, sidx] = sort([L.time]);
                L = L(sidx);
            end
        end
    end

end