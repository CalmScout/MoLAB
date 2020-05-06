% Load all .mat files from the folder and save them again - in the newest
% format, otherwise Python libs nmay cause errors during `.mat` file loading.

% folder = '/storage/dataset_classification/brain_metastasis/mat/';
% folder_to = '/storage/dataset_classification/brain_metastasis/mat_v7/';

%folder = '/storage/dataset_classification/meningiomas/mat/'
%folder_to = '/storage/dataset_classification/meningiomas/mat_v7/'

folder = '/storage/dataset_classification/high_grade_glioma/mat/'
folder_to = '/storage/dataset_classification/high_grade_glioma/mat_v7/'

my_files = dir(fullfile(folder, '*.mat'));
for k = 1:length(my_files)
    file_name = my_files(k).name;
    full_file_name = fullfile(folder, file_name);
    fprintf('File %d / %d \n', k, length(my_files))
    fprintf(1, 'Reading %s\n', full_file_name);
    data = load(full_file_name);
    disp(data)
    full_file_name_to = fullfile(folder_to, file_name);
%     save(full_file_name_to, 'data', '-v7.3');
    save(full_file_name_to, 'data', '-v7');
    fprintf(1, 'Saved to the disk %s\n', full_file_name_to);
end
