classdef ThreadIdx < handle
    properties
        startidx
        endidx
        curr
    end



    methods(Static)

        function [s, e] = threadBounds(N, Nthreads)
            nt = floor(N/Nthreads);
            idx = 1:Nthreads;
            s = (idx-1)*nt + 1;
            e = (idx)*nt;
            e(end) = N;
        end

    end

    methods
        function obj = ThreadIdx(N, Nthreads)
            [obj.startidx, obj.endidx] = ThreadIdx.threadBounds(N, Nthreads);
            obj.curr = obj.startidx;
        end

        function idx = next(obj, tid)
            idx = obj.curr(tid);
            obj.curr(tid) = idx + 1;
        end

        function c = complete(obj,tid)
            c = obj.curr(tid) == obj.endidx(tid) + 1;
        end

    end
end