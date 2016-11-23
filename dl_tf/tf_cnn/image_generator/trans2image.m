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
    case '2DFFT'
end

end