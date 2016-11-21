function Z = trans2image(f,Fs)


fName = fieldnames(f);
% Fs = f.(fName{1}).iEEGsamplingRate;     % Sampling Freq
eegData = f.(fName{1}).data(1:Fs:end,:);% EEG data matrix, resampled at Fs
[nt,nc] = size(eegData);

Z = zeros(nt, nt, nc);

for n = 1:nc
    ydft = fft(eegData(:,n));
    for i = 1:nt
        for j = 1:nt
            Z(i,j,n) = exp(-abs(ydft(i))/abs(ydft(j)));
        end
    end    
end




end