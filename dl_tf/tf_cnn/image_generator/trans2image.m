function Z = trans2image(in_data, ts_fs, fft_fs,transform_type)
% transform_type : 'rings', 'parcheh' or '2DFFT'
% ts_fs: time series sampling time
% fft_fs: fft sampling time

fName = fieldnames(in_data);
% Fs = f.(fName{1}).iEEGsamplingRate;     % Sampling Freq
eegData = in_data.(fName{1}).data(1:ts_fs:end,:);% EEG data matrix, resampled at Fs

% check if the data is zero
if sum(eegData(:)) == 0
    Z = 0;
    return
end

[nt,nc] = size(eegData);

switch transform_type
    case 'rings'
        
    case 'resp_ffts'
        ydft = fft(eegData(:,1));
        len = length(ydft(1:fft_fs:end));
        Z = zeros(len,len,nc);
        for n = 1:nc
            ydft = fft(eegData(:,n));
            ydft = fftshift(ydft);
            ydft = ydft(1:fft_fs:end);
            for i = 1:len
                for j = 1:len
                    Z(i,j,n) = exp(-abs(ydft(i))/(1 + abs(ydft(j))));
                end
            end            
        end
%         for i = 1:16
%             subplot(4,4,i);
%             imshow(Z(:,:,i));
%         end
%         keyboard;
    case 'rep_fft'
        y = fft(eegData);
        F = y(1:fft_fs:end,:);
        S = fftshift(F);
        S = kron(S, ones(1,length(F) / nc));
        A = abs(S);
        Z = zeros(length(A),length(A),2);
        Z(:,:,1) = A / max(A(:));
        Z(:,:,2) = Z(:,:,1)';
end

end