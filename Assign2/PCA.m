clear all;
close all;
clc;

%% LOAD IMAGES AND LABELS.
fprintf('Loading images and labels.\n');
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
display_network(images(:,1:256)); % Show the first 256 images

[n, M] = size(images);

%% GENERATING AND DISPLAYING MEAN IMAGE.
mean_image = zeros(784, 1);
for i=1:M
    mean_image = mean_image + images(:, i)/M;
    %     if(mod(i, 100)==0)
    %         display_network(meanimg(:,1));
    %     end
end
display_network(mean_image(:,1)); % Show the mean image

%% CENTERING IMAGES BY SUBTRACTING MEAN IMAGE.
centred_images = zeros(n, M);
for i=1:M
    centred_images(:, i) = images(:, i) - mean_image;
end
display_network(centred_images(:,1:256)); % Show the first 256 images after subtracting mean

%% USING COVARIANCE MATRIX TO GET EIGEN VALUES AND VECTORS.
XXT = 1/M*(centred_images*centred_images');
[eig_vec, eig_val] = eigs(XXT, n);

%% CHOOSING NUMBER OF PCs.
k = 25;

%% PROJECTING DATASET.
projection = eig_vec(:, 1:k)'*centred_images;

%% REGENERATING IMAGES FROM FIRST K COMPONENTS.
regenerated = eig_vec(:, 1:k)*projection;
for i=1:M
    regenerated(:, i) = regenerated(:, i) + mean_image;
end
display_network(regenerated(:, 1:256)); % Show the first 256 images after recovery
