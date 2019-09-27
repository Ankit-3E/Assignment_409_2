function DataGen(entry,images,labels,Imind)
%     fprintf('Loading images and labels.\n');
%     images = loadMNISTImages('train-images.idx3-ubyte');
%     labels = loadMNISTLabels('train-labels.idx1-ubyte');
    [n, M] = size(images);
    mean_image = zeros(784, 1);
    for i=1:M
        mean_image = mean_image + images(:, i)/M;
    end
    centred_images = zeros(n, M);
    for i=1:M
        centred_images(:, i) = images(:, i) - mean_image;
    end
%     centred_images = images;
    XXT = 1/M*(centred_images*centred_images');
    [eig_vec, eig_val] = eig(XXT);
    [deig,ind] = sort(diag(eig_val),'descend');
%     deig_val = eig_val(ind,ind);
    deig_vec = eig_vec(:,ind);
    k = 25;
    
    projection = deig_vec(:, 1:k)'*centred_images;
    data = projection';
    data = [data labels];
%     images = images';
%     images = [images labels];
    Imind = [Imind labels];
    path = strcat('A2/PCA/',strcat(char(entry),'.csv'));
    pathImg = strcat('A2/Images/',strcat(char(entry),'.csv'));
    csvwrite(path,data);
    csvwrite(pathImg,Imind);
end
