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

        audioIdx
        spectroIdx

    end

    methods
        function obj = DataLoader(ds, mode)
            if nargin < 2
                mode = "single";
            end
            obj.ds = ds;
            obj.audiolist = obj.listAudio();
            obj.spectrogramlist = obj.listSpectrograms();
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

        function [D, info] = nextAudio(obj, tid)
            if nargin < 2
                tid = 1;
            end
            idx = obj.audioIdx.next(tid);
            [D, info] = obj.loadAudio(idx);
        end

        function [s, info] = nextSpectrogram(obj, tid)
            if nargin < 2
                tid = 1;
            end
            idx = obj.spectroIdx.next(tid);
            [s, info] = obj.loadSpectrogram(idx);
        end

        function c = isAudioComplete(obj, tid)
            c = obj.audioIdx.complete(tid);
        end

        function c = isSpectrogramComplete(obj, tid)
            c = obj.spectroIdx.complete(tid);
        end

    end

    methods(Access=private)
        function list = listAudio(obj)
            p = sprintf("%s\\%d", DataLoader.audiopath, obj.ds);
            files = dir(p);
            files = files(~[files.isdir]);
            parfor i = 1:numel(files)
                name = string(files(i).name);
                path = sprintf("%s\\%s", p, name);
                time = datetime(extractBetween(name, 1, 8+6+4+2), ...
                    "InputFormat","yyyyMMdd_HHmmss_SSSS");
                list(i) = struct("name", name, "path", path, "time", time);
            end
            [~, sidx] = sort([list.time]);
            list = list(sidx);
        end

        function list = listSpectrograms(obj)
            p = sprintf("%s\\%d", DataLoader.spectrogrampath, obj.ds);
            files = dir(p);
            segT = seconds(2^18/2000);
            files = files(~[files.isdir]);
            parfor i = 1:numel(files)
                name = string(files(i).name);
                path = sprintf("%s\\%s", p, name);
                time = datetime(extractBetween(name, 1, 8+6+4+2), ...
                    "InputFormat","yyyyMMdd_HHmmss_SSSS");
                seg = regexp( extractAfter(name, 8+6+4+2), '\d+', ...
                    'match', 'once' );
                seg = str2double(seg);
                time = time + (seg-1)*segT;
                list(i) = struct("name", name, "path", path, "time", time);
            end
            [~, sidx] = sort([list.time]);
            list = list(sidx);
        end
    end

end