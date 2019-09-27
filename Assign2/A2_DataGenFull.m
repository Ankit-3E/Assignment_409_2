clear all;
close all;
clc;

%% LOAD IMAGES AND LABELS.
fprintf('Loading images and labels.\n');
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
entryTable = readtable('entry_num.csv');

for i= 1:size(entryTable)
    imgs = zeros(784,0);
    lab = zeros(1,0);
    entry_num = entryTable{i,1};
    for j=0:9
        ind = find(labels==j);
        rand_ind = randperm(size(ind,1),300);
        ind = ind(rand_ind);
        imgs = [imgs images(:,ind)];
        lab = [lab; labels(ind)];
    end
    random_indices = randperm(3000);
    imgs = imgs(:,random_indices);
    lab = lab(random_indices);
    DataGen(entry_num,imgs,lab);
end