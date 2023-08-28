%these are the only datasets that are usable
outputnames = [...
    "BallenyIslands2015",...
    "CaseyIslands2017",...
    "KerguelenIslands2015",...
    ];

inputdirs = [...
    "D:\\Whale Data\\AcousticTrends_BlueFinLibrary\\BallenyIslands2015\\",...
    "D:\\Whale Data\\AcousticTrends_BlueFinLibrary\\casey2017\\",...
    "D:\\Whale Data\\AcousticTrends_BlueFinLibrary\\kerguelen2015\\",
    ];

outputdir = "D:\\Whale Data\\Raw Audio Data\\";

dtformat = "yyyyMMdd_HHmmss";

Q = 16;
T = 1.5;
fs = 1000;
flow = 15;
fhigh = 120;

% %create spectrograms
% for idx = 2%numel(outputnames)
%     sc = SpectrogramCreator(outputnames(idx));
%     sc.create(Q, T, fs, AudioDataConverter.L, flow, fhigh);
% end
%create spectrograms
% for idx = 2%numel(outputnames)
%     sc = STFTCreator(outputnames(idx));
%     sc.create(512, 256, fs);
% end
% 
% %create gmms
% for idx = 2%numel(outputnames)
%     nec = NoiseEstimateCreator(outputnames(idx));
%     nec.create();
% end
% 


T = 1.5;
Q = [16, 4];
flow = 15;
fhigh = 120;
fs = 250;
oversample = 2;

% %scattering
% for idx = 2%numel(outputnames)
%     scc = ScatteringCreator(outputnames(idx), Q, flow, fhigh, T, oversample);
%     scc.create();
% end

%convert annotations
for fidx = 2%numel(outputnames)
    ac = AnnotationConverter(inputdirs(fidx), outputnames(fidx), 1000);
    ac.convert();
end