load('/home/n2mohaje/Code/rnnSeizureDetection/data_dir/Kaggle_data/data/train_1/1_1_0.mat');
fs = 1000;
% for i = 1:16
I = dataStruct.data;
F = fft2(I);
S = fftshift(F);
L=log2(S);
% L = repmat(L(1:fs:end,:));
A=abs(L);
% subplot(16,1,1)
% imagesc(I');
% subplot(4,4,i)
figure;
imagesc(I');
colormap gray;
figure;
colormap gray;
imagesc(A');
% end