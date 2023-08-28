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
        scatteringfs
        xfs
        f
    end

    methods

        function this = AnnotationConverter(inputpath, outputname, xfs)
            this.inputpath = inputpath;
            this.outputname = outputname;
            this.outputpath = AnnotationConverter.datapath + this.outputname;
            fb = load(this.outputpath + "\\scattering\\filterbank.mat").fb;
            this.scatteringfs = fb.filterBanks(1).getSSamplingFreq();
            this.xfs = xfs;
            this.f = fb.filterBanks(1).fc;
        end

        function convert(this)

            dl = DataLoader2(this.outputpath, 'wav', 'single', true);
            filelist = dl.filelist;
            FIDlist = [filelist.fid];
            [filelisttimes, sidx] = sort([filelist.time]);
            filelist = filelist(sidx);
            FIDlist = FIDlist(sidx);

            minf = min(this.f);
            maxf = max(this.f);

            files = dir(this.inputpath);
            annotations = convertCharsToStrings({files.name});
            idx = arrayfun(@(s) s.contains("selections.txt"), annotations);
            annotations = annotations(idx);
            ann = {};
            for fname = annotations
                T = readtable(this.inputpath + fname, "Delimiter","\t");
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
                    dt = seconds(row.StartIndex/this.xfs);
                    ftime = ftime + dt;
                    matchedfileidx = arrayfun(@(time)find(time >= (filelisttimes'), 1, "last"), ftime);
                    matchedfile = [filelist(matchedfileidx).name]';
                    matchedfiletimes = [filelist(matchedfileidx).time]';
                    matchedfids = [FIDlist(matchedfileidx)]';
                    offset = floor(seconds(ftime - matchedfiletimes)*this.xfs);
                    dur = ceil(row.Duration * this.xfs);
                    row.StartIndex = offset;
                    row.EndIndex = row.StartIndex + dur;
                    row.OrigFile = row.File;
                    row.File = matchedfile;
                    row.FileID = matchedfids;

                    %scattering
                    specstart = floor(row.StartIndex / this.xfs * this.scatteringfs) + 1; %covert bins to sampling freq of the spectrogram
                    specend = ceil(row.EndIndex / this.xfs * this.scatteringfs) + 1;
                    row.ScatteringStartIndex = specstart;
                    row.ScatteringEndIndex = specend;

                    row.AnnotationToNoiseRatio = zeros(size(specstart));
                    row.AnnotationToNoiseRatioDB = zeros(size(specstart));
                    row.AnnotationSignificance = zeros(size(specstart));


                    annotationTable{end + 1} = row;
                end
            end
            this.annotationTable = vertcat(annotationTable{:});
            this.annotationTable.AnnotationID = (1:size(this.annotationTable, 1))';
            %clean up annotation frequencies
            this.annotationTable = this.annotationTable(this.annotationTable.StartFrequency < maxf, :);
            this.annotationTable.StartFrequency = max(this.annotationTable.StartFrequency, minf);
            this.annotationTable.EndFrequency = min(this.annotationTable.EndFrequency, maxf);
%             this.calcANR();

            savepath = this.outputpath + "\\annotations";
            mkdir(savepath);
            annotationTable = this.annotationTable;
            save(savepath + "\\annotations.mat", "annotationTable", '-mat');
        end

        function calcANR(this)
            fileIDs = unique(this.annotationTable.FileID);
            specpath = this.outputpath + "\\spectrograms\\";
            noisepath = this.outputpath + "\\noise\\NoiseEstimate.mat";
            varpath = this.outputpath + "\\noise\\NoiseVarianceEstimate.mat";
            noiseEstimate = load(noisepath).noiseEstimates;
            noiseVarianceEstimate = load(varpath).noiseVarianceEstimates;
            for idx = 1:numel(fileIDs)
                %get spectrogram
                fid = fileIDs(idx);
                annidx = this.annotationTable.FileID == fid;
                annotations = this.annotationTable(annidx, :);
                specname = extractBefore(annotations.File(1), ".wav") + ".mat";
                s = load(specpath + specname).s;
                s = s.^2;
                Snoise = noiseEstimate(:, fid);
                Svar = noiseVarianceEstimate(:, fid);
                %get file and limit EndIndex for annotations that overlap
                %with files
                rowidx = 1:size(this.annotationTable, 1);
                rowidx = rowidx(annidx);
                this.annotationTable.SpectrogramEndIndex(rowidx) = min(this.annotationTable.SpectrogramEndIndex(rowidx), size(s, 2));
                annotations = this.annotationTable(annidx, :);
                %calculate ANR
                for j = 1:size(annotations, 1)
                    row = annotations(j, :);
                    ridx = rowidx(j);
                    sidx = row.SpectrogramStartIndex;
                    eidx = row.SpectrogramEndIndex;
                    [~, fstart] = min(abs(this.f - row.StartFrequency));
                    [~, fend] = min(abs(this.f - row.EndFrequency));
                    try
                    meanpower = mean(sum(s(fstart:fend, sidx:eidx), 1));
                    noisepower = sum(Snoise(fstart:fend));
                    catch
                        disp 1            
                    end
                    noisevar = sum(Svar(fstart:fend));
                    this.annotationTable.AnnotationToNoiseRatio(ridx) = (meanpower - noisepower)/noisepower;
                    this.annotationTable.AnnotationToNoiseRatioDB(ridx) = mag2db((meanpower - noisepower)/noisepower);
                    this.annotationTable.AnnotationSignificance(ridx) = 1 - normcdf(meanpower, noisepower, sqrt(noisevar)); %p-value for a normal distribution
                end
            end
        end


    end


end