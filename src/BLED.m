classdef BLED < handle
    properties(Constant)
    end

    properties
        probs
    end

    methods
        function detect(obj, S)
            e = sum(S, 1);
            obj.probs = e;
        end
    end

end