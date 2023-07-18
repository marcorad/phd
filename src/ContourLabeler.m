classdef ContourLabeler < handle

    properties(Constant)
        datapath = "D:\\Whale Data\\Raw Audio Data\\";
    end

    properties
        path
        annpath
        contourpath
        foldername
        annotationTable
        contourTable
        sfs
        xfs
        AnnotationCount
        AnnotationDetected
    end

    methods
        function obj = ContourLabeler(foldername, sfs, xfs, sig)
            obj.foldername = foldername;
            obj.path = SpectrogramCreator.datapath + string(foldername);
            obj.annpath = obj.path + "\\annotations\\";
            obj.contourpath = obj.path + "\\contours\\";

            obj.annotationTable = load(obj.annpath + "annotations.mat").annotationTable;
            obj.annotationTable = obj.annotationTable(obj.annotationTable.AnnotationSignificance <= sig, :);
            obj.contourTable = load(obj.contourpath + "ContourTable.mat").featureTable;

            obj.sfs = sfs;
            obj.xfs = xfs;
        end

        function match(obj)
            %match the annotations
            annotationFIDs = unique(obj.annotationTable.FileID);
            sarr = strings(size(obj.contourTable, 1), 1);
            sarr(:) = "Unmatched";
            obj.contourTable.Annotation = sarr;
            obj.contourTable.AnnotationID = ones(size(obj.contourTable, 1), 1)*-1;
            obj.contourTable.TimeOverlap = zeros(size(obj.contourTable, 1), 1);
            obj.AnnotationCount = zeros(size(obj.annotationTable, 1), 1);
            for fididx = 1:numel(annotationFIDs)
                %get all the contours matching the current annotation File
                %ID
                if mod(fididx, 100) == 0
                    disp(fididx/numel(annotationFIDs) * 100)
                end
                fid = annotationFIDs(fididx);
                contourMask = obj.contourTable.FileID == fid;
                annotationMask = obj.annotationTable.FileID == fid;
                annotationIdx = find(annotationMask);
                contours = obj.contourTable(contourMask, :);
                annotations = obj.annotationTable(annotationMask, :);
                for i = 1:size(contours, 1) %for each contour, check if it matches an annotation
                    c = contours(i, :);
                    o = obj.overlaps(c, annotations);
                    matches = annotations(o, :);
                    [A, toverlap] = obj.overlapArea(c, matches);
                    if size(matches, 1) > 1 %multiple matches, so choose one with most overlap area                        
                        [~, maxidx] = max(A, [], "all");
                        matches = matches(maxidx, :);
                        toverlap = toverlap(maxidx);
                        obj.AnnotationCount(annotationIdx(maxidx)) = obj.AnnotationCount(annotationIdx(maxidx)) + 1;
                    end
                    if size(matches, 1) > 0
                        contours(i, :).Annotation = matches(1, :).Annotation;
                        contours(i, :).AnnotationID = matches(1, :).AnnotationID;
                        contours.TimeOverlap(i) = toverlap;
                    end
                end
                obj.contourTable(contourMask, :) = contours;
            end

            obj.AnnotationDetected = obj.AnnotationCount > 0;
            obj.annotationTable.Detected = obj.AnnotationDetected;

            %clean up the matches,as multiple contours may be assigned per
            %anotation
%             annid = unique(obj.contourTable.AnnotationID);
%             annid = annid(annid ~= -1);
%             for idx = 1:numel(annid)
%                 matchidx = obj.contourTable.AnnotationID == annid(idx);
%                 matchrows = find(matchidx);
%                 matches = obj.contourTable(matchidx, :);
%                 [~, maxoverlapidx] = max(matches.TimeOverlap, [], "all");
%                 select = matchrows(maxoverlapidx);
%                 removeidx = matchidx;
%                 removeidx(select) = false;
%                 obj.contourTable.Annotation(removeidx) = "Unmatched";
%             end


            T = obj.contourTable;
            save(obj.annpath + "MatchedAnnotations.mat", "T", '-mat');
        end

        function [o] = overlaps(obj, c, a)
            o = (c.StartIndex <= a.SpectrogramEndIndex) & (c.EndIndex >= a.SpectrogramStartIndex);
            o = o & (c.Frequency(3) <= a.EndFrequency) & (c.Frequency(6) >= a.StartFrequency);
        end

        function [A, timeOverlap] = overlapArea(obj, c, a)
            dx = (min(c.EndIndex, a.SpectrogramEndIndex) - max(c.StartIndex, a.SpectrogramStartIndex));
            dy = (min(c.Frequency(6), a.EndFrequency) - max(c.Frequency(3), a.StartFrequency));
            A =  dx .* dy;
            timeOverlap = dx./(a.SpectrogramEndIndex - a.SpectrogramStartIndex);
        end


    end

end