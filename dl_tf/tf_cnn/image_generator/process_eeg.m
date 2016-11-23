function process_eeg(subjectNames,segmentTypes,options)

%
%% Options
fopts = fieldnames(options);

if sum(strcmp(fopts,'dataDir'))~=0
    dataDir = options.dataDir;
else
    error('Please specify the input data location.')
end

if sum(strcmp(fopts,'featureDir'))~=0
    featureDir = options.featureDir;
else
    error('Please specify the output feature location.')
end

if sum(strcmp(fopts,'clustername'))~=0
    clustername = options.clustername;
else
    clustername = 'local';
end

if ~strcmp(clustername,'local')
    if sum(strcmp(fopts,'N'))~=0
        N = options.N;
    else
        error('Please specify preferred number of workers in a parallel pool.')
    end
end

if ~strcmp(clustername,'local')
    if ~strcmp(fopts,'numChunk')
        error('Please specify the number of data chunks.')
    else
        numChunk    = options.numChunk;
    end
end

if ~iscell(subjectNames)
    subjectNames = cellstr(subjectNames);
end

if isempty(segmentTypes) % meaning that it's a test data segment
    segmentTypes = {'*'};
elseif ~iscell(segmentTypes)
    error('Input must be a cell string. Use num2str for conversion.')
end


%% Read data, compute and save features
for i = 1:length(subjectNames)
    %for j = 1:length(segmentTypes)

        % Specify patient to look at
        subjectName = subjectNames{i};
        % Specify segment type
        %segmentType = segmentTypes{j};

        % Read and count number of files associated with this segment type
        sourceDir = [dataDir filesep subjectName];
        
        %fileNames = dir([sourceDir filesep '*' segmentType '.mat']);
        fileNames = dir([sourceDir filesep '*' '.mat']);
        numFiles = length(fileNames);
        FileNames = {fileNames(:).name};
        FilePaths = fullfile(dataDir, subjectName, FileNames);
        savePath = fullfile(featureDir, subjectName);
        %if ~isdir(savePath)
        %    mkdir(savePath);
        %end
        %% Calculate features
        for k = 1:numFiles

            % Load and display the file being read.
            fileName = strrep(fileNames(k).name,'.mat','');
            filePath = fullfile(dataDir, subjectName, fileNames(k).name);
            f = load(filePath);
%             disp(filePath);
            
            eeg_image = trans2image(f, ...
                options.ts_sampling, ...
                options.fft_sampling, ...
                options.transform_type);
            if eeg_image == 0
                disp(['All zeros detected: ' filePath]);
            else
                save(['../../../data_dir/Kaggle_data/data/image_train_3_300/resp_ffts/' fileName], 'eeg_image');
            end

            % Calculate features
            %feat = calculate_features(f);
            

            % Store features to featureDir
            %parsave([savePath filesep fileName],'feat',feat);
        end
        disp(['Done. Saved all images to ' savePath])

    %end
end

end

function parsave(filepath, varStr, var)
evalc([varStr '=' 'var']);
save(filepath, varStr);
end

