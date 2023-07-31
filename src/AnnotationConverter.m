classdef AnnotationConverter < handle

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
        dtformat = "yyyyMMdd_HHmmss";
    end

    properties
        inputpath
        outputname
        outputpath
        annotationTable
        scalogramfs
        scatteringfs
        xfs
        f
    end

    methods

        function obj = AnnotationConverter(inputpath, outputname)
            obj.inputpath = inputpath;
            obj.outputname = outputname;
            obj.outputpath = AnnotationConverter.datapath + obj.outputname;
            fb = load(obj.outputpath + "\\spectrograms\\filterbank.mat").fb;
            obj.scalogramfs = fb.getSSamplingFreq();
            fb = load(obj.outputpath + "\\scattering\\filterbank.mat").fb;
            obj.scatteringfs = fb.filterBanks(1).getSSamplingFreq();
            obj.xfs = fb.filterBanks(1).fs;
            obj.f = fb.filterBanks(1).fc;
        end

        function convert(obj)

            dl = DataLoader2(obj.outputpath, 'wav', 'single', true);
            filelist = dl.filelist;
            FIDlist = [filelist.fid];
            [filelisttimes, sidx] = sort([filelist.time]);
            filelist = filelist(sidx);
            FIDlist = FIDlist(sidx);

            minf = min(obj.f);
            maxf = max(obj.f);

            files = dir(obj.inputpath);
            annotations = convertCharsToStrings({files.name});
            idx = arrayfun(@(s) s.contains("selections.txt"), annotations);
            annotations = annotations(idx);
            ann = {};
            for fname = annotations
                T = readtable(obj.inputpath + fname, "Delimiter","\t");
                idx = find(fname{:} == '.');
                type = extractBetween(fname, idx(1)+1, idx(end-1)-1);
                type = type.replace(".", "_");
                type = repmat(type, size(T, 1), 1);
                T(:, 'Type') = cellstr(type);
                ann{end + 1} = T;
            end
            annotationTable = {};
            for T = ann
                T = T{:};
                row = table();
                row.File = convertCharsToStrings(T.BeginFile);
                if ~isempty(row)
                    
                    if T.LowFreq_Hz_ >= maxf
                        continue;
                    end

                    row.StartIndex = T.BegFileSamp_samples_;
                    row.EndIndex = T.EndFileSamp_samples_;
                    row.Duration = T.DeltaTime_s_;
                    row.Annotation = convertCharsToStrings(T.Type);

                    row.StartFrequency = T.LowFreq_Hz_;
                    row.EndFrequency = T.HighFreq_Hz_;


                    %match the correct file
                    timestr = extractBefore(lower(row.File), ".wav");
                    timestr = extractBetween(timestr, 1, strlength(AnnotationConverter.dtformat));
                    ftime = datetime(timestr, "InputFormat", AnnotationConverter.dtformat);
                    dt = seconds(row.StartIndex/1000);
                    ftime = ftime + dt;
                    matchedfileidx = arrayfun(@(time)find(time >= (filelisttimes'), 1, "last"), ftime);
                    matchedfile = [filelist(matchedfileidx).name]';
                    matchedfiletimes = [filelist(matchedfileidx).time]';
                    matchedfids = [FIDlist(matchedfileidx)]';
                    offset = floor(seconds(ftime - matchedfiletimes)*1000);
                    dur = ceil(row.Duration * 1000);
                    row.StartIndex = offset;
                    row.EndIndex = row.StartIndex + dur;
                    row.OrigFile = row.File;
                    row.File = matchedfile;
                    row.FileID = matchedfids;

                    %scalogram
                    specstart = floor(row.StartIndex / obj.xfs * obj.scalogramfs) + 1; %covert bins to sampling freq of the spectrogram
                    specend = ceil(row.EndIndex / obj.xfs * obj.scalogramfs) + 1;
                    row.ScalogramStartIndex = specstart;
                    row.ScalogramEndIndex = specend;
                    %scattering
                    specstart = floor(row.StartIndex / obj.xfs * obj.scatteringfs) + 1; %covert bins to sampling freq of the spectrogram
                    specend = ceil(row.EndIndex / obj.xfs * obj.scatteringfs) + 1;
                    row.ScatteringStartIndex = specstart;
                    row.ScatteringEndIndex = specend;

                    row.AnnotationToNoiseRatio = zeros(size(specstart));
                    row.AnnotationToNoiseRatioDB = zeros(size(specstart));
                    row.AnnotationSignificance = zeros(size(specstart));


                    annotationTable{end + 1} = row;
                end
            end
            obj.annotationTable = vertcat(annotationTable{:});
            obj.annotationTable.AnnotationID = (1:size(obj.annotationTable, 1))';
            %clean up annotation frequencies
            obj.annotationTable = obj.annotationTable(obj.annotationTable.StartFrequency < maxf, :);
            obj.annotationTable.StartFrequency = max(obj.annotationTable.StartFrequency, minf);
            obj.annotationTable.EndFrequency = min(obj.annotationTable.EndFrequency, maxf);
%             obj.calcANR();

            savepath = obj.outputpath + "\\annotations";
            mkdir(savepath);
            annotationTable = obj.annotationTable;
            save(savepath + "\\annotations.mat", "annotationTable", '-mat');
        end

        function calcANR(obj)
            fileIDs = unique(obj.annotationTable.FileID);
            specpath = obj.outputpath + "\\spectrograms\\";
            noisepath = obj.outputpath + "\\noise\\NoiseEstimate.mat";
            varpath = obj.outputpath + "\\noise\\NoiseVarianceEstimate.mat";
            noiseEstimate = load(noisepath).noiseEstimates;
            noiseVarianceEstimate = load(varpath).noiseVarianceEstimates;
            for idx = 1:numel(fileIDs)
                %get spectrogram
                fid = fileIDs(idx);
                annidx = obj.annotationTable.FileID == fid;
                annotations = obj.annotationTable(annidx, :);
                specname = extractBefore(annotations.File(1), ".wav") + ".mat";
                s = load(specpath + specname).s;
                s = s.^2;
                Snoise = noiseEstimate(:, fid);
                Svar = noiseVarianceEstimate(:, fid);
                %get file and limit EndIndex for annotations that overlap
                %with files
                rowidx = 1:size(obj.annotationTable, 1);
                rowidx = rowidx(annidx);
                obj.annotationTable.SpectrogramEndIndex(rowidx) = min(obj.annotationTable.SpectrogramEndIndex(rowidx), size(s, 2));
                annotations = obj.annotationTable(annidx, :);
                %calculate ANR
                for j = 1:size(annotations, 1)
                    row = annotations(j, :);
                    ridx = rowidx(j);
                    sidx = row.SpectrogramStartIndex;
                    eidx = row.SpectrogramEndIndex;
                    [~, fstart] = min(abs(obj.f - row.StartFrequency));
                    [~, fend] = min(abs(obj.f - row.EndFrequency));
                    try
                    meanpower = mean(sum(s(fstart:fend, sidx:eidx), 1));
                    noisepower = sum(Snoise(fstart:fend));
                    catch
                        disp 1            
                    end
                    noisevar = sum(Svar(fstart:fend));
                    obj.annotationTable.AnnotationToNoiseRatio(ridx) = (meanpower - noisepower)/noisepower;
                    obj.annotationTable.AnnotationToNoiseRatioDB(ridx) = mag2db((meanpower - noisepower)/noisepower);
                    obj.annotationTable.AnnotationSignificance(ridx) = 1 - normcdf(meanpower, noisepower, sqrt(noisevar)); %p-value for a normal distribution
                end
            end
        end


    end


end