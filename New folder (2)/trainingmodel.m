clc
clear all
close all
warning off
g=alexnet;
layer=g.Layers;
layers(23)=fullyConnectedLayer(2);
layers(25)=classificationLayer;
allImageDatastore('datastorage','IncludeSubfolders',true,'LabelSource','foldernames');
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
myNet1=trainNetwork(allImage,layers,opts);
save myNet1;
