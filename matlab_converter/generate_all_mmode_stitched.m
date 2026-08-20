%% =====================================================================
%% generate_all_mmode_stitched.m
%% Generates continuous horizontally-stitched M-Mode (Time-Motion) echograms
%% for all .bin files in E:\Probefix\britty (5rm and 10rm).
%% Outputs medical grayscale (DR = 78 dB) images for full-session inspection.
%% =====================================================================

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
parent_dir = fileparts(project_root);

britty_dir = fullfile(parent_dir, 'britty');
output_dir = fullfile(parent_dir, 'dataset', 'mmode_stitched');

addpath(script_dir);

conditions = {'5rm', '10rm'};
dr_dB = 78;
center_col = 17; % 1-based index (center of 33 scanlines)

for c_idx = 1:length(conditions)
    cond_name = conditions{c_idx};
    cond_path = fullfile(britty_dir, cond_name);
    cond_out_dir = fullfile(output_dir, cond_name);
    
    if ~exist(cond_out_dir, 'dir')
        mkdir(cond_out_dir);
    end
    
    sub_dirs = dir(cond_path);
    sub_dirs = sub_dirs([sub_dirs.isdir]);
    sub_dirs = sub_dirs(~ismember({sub_dirs.name}, {'.', '..'}));
    
    fprintf('\nProcessing Condition: [%s]\n', cond_name);
    take_id = 0;
    
    for s_idx = 1:length(sub_dirs)
        sub_folder = sub_dirs(s_idx).name;
        if ~contains(lower(sub_folder), 'open'), continue; end
        
        current_dir = fullfile(cond_path, sub_folder);
        bin_files = dir(fullfile(current_dir, '*.bin'));
        
        for f_idx = 1:length(bin_files)
            take_id = take_id + 1;
            file_name = bin_files(f_idx).name;
            [~, raw_name, ~] = fileparts(file_name);
            
            fprintf('  [Take %02d] Stitching: %s/%s ...\n', take_id, sub_folder, file_name);
            
            [RF_DATA, ~, HEADER, ~, flag] = import_RFdata2MATLAB([current_dir, filesep], file_name);
            if flag == 0 || isempty(RF_DATA), continue; end
            
            num_frames = size(RF_DATA, 2);
            num_samples = HEADER{1}.Length_of_RF_row;
            
            m_mode_matrix = zeros(num_samples, num_frames);
            for t = 1:num_frames
                rf_cell = RF_DATA{1, t};
                if ~isempty(rf_cell)
                    m_mode_matrix(:, t) = abs(double(rf_cell(:, center_col)));
                end
            end
            
            % Logarithmic dynamic range compression (DR = 78 dB)
            env_max = max(m_mode_matrix(:));
            if env_max == 0, env_max = 1; end
            env_norm = m_mode_matrix / env_max;
            env_dB = 20 * log10(env_norm + eps);
            bmode_mmode = (env_dB + dr_dB) / dr_dB;
            bmode_mmode = max(0, min(1, bmode_mmode));
            
            out_name = sprintf('mmode_%s_take_%02d_%s.png', cond_name, take_id, raw_name);
            out_file = fullfile(cond_out_dir, out_name);
            imwrite(uint8(bmode_mmode * 255), out_file);
            
            fprintf('      -> Saved: %s (Size: %d depth x %d frames)\n', out_name, num_samples, num_frames);
        end
    end
end

fprintf('\nAll M-Mode stitched files generated in: %s\n', output_dir);
