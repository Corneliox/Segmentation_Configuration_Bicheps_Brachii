%% =====================================================================
%% convert_bin2dataset.m
%% Converts ArtUs / Telemed .bin RF ultrasound data to 8-bit Grayscale PNGs
%% with EchoWave-matched Logarithmic Dynamic Range Compression (DR = 78 dB)
%% =====================================================================

clc; clear; close all;

% Directories relative to this script
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
parent_dir = fileparts(project_root);

raw_bin_dir = fullfile(parent_dir, 'BIN FILE');
output_img_dir = fullfile(parent_dir, 'dataset', 'images');
output_mask_dir = fullfile(parent_dir, 'dataset', 'masks');

% Ensure output directories exist
if ~exist(output_img_dir, 'dir')
    mkdir(output_img_dir);
end
if ~exist(output_mask_dir, 'dir')
    mkdir(output_mask_dir);
end

% Path to existing import_RFdata2MATLAB function
import_func_dir = fullfile(parent_dir, 'Ultrasound_Gilang', '1_Convert');
addpath(import_func_dir);

% List all .bin files in raw_bin_dir
bin_files = dir(fullfile(raw_bin_dir, '*.bin'));
fprintf('Found %d .bin files in %s\n', length(bin_files), raw_bin_dir);

% Parameters matching EchoWave Screenshot (31)
dr_dB = 78;          % Dynamic Range in dB (78 dB)
gauss_sigma = 0.8;   % Filter sigma for speckle reduction

frame_counter = 0;

for f_idx = 1:length(bin_files)
    file_name = bin_files(f_idx).name;
    file_path = fullfile(raw_bin_dir, file_name);
    [~, clean_name, ~] = fileparts(file_name);
    clean_name = regexprep(clean_name, '[^a-zA-Z0-9_-]', '_');
    
    fprintf('Processing [%d/%d]: %s ...\n', f_idx, length(bin_files), file_name);
    
    try
        [RF_DATA, RF_DATA_Q, HEADER, transducer_code, flag] = import_RFdata2MATLAB(raw_bin_dir, file_name);
        
        num_slices = size(RF_DATA, 2);
        
        for slice_idx = 1:num_slices
            rf_cell = RF_DATA{1, slice_idx};
            if isempty(rf_cell)
                continue;
            end
            
            % 1. Envelope Detection (Demodulation)
            envelope = abs(hilbert(double(rf_cell)));
            
            % 2. Logarithmic Dynamic Range Compression
            env_max = max(envelope(:));
            if env_max == 0
                env_max = 1;
            end
            env_norm = envelope / env_max;
            env_dB = 20 * log10(env_norm + eps);
            bmode = (env_dB + dr_dB) / dr_dB;
            bmode(bmode < 0) = 0;
            bmode(bmode > 1) = 1;
            
            % 3. Gaussian Filtering for speckle smoothing
            bmode_filtered = imgaussfilt(bmode, gauss_sigma);
            
            % 4. Save as 8-bit PNG grayscale image
            out_filename = sprintf('img_%s_slice_%04d.png', clean_name, slice_idx);
            out_path = fullfile(output_img_dir, out_filename);
            
            imwrite(uint8(bmode_filtered * 255), out_path);
            frame_counter = frame_counter + 1;
        end
    catch ME
        fprintf('  [Warning] Failed to convert %s: %s\n', file_name, ME.message);
    end
end

fprintf('\nSuccessfully extracted %d grayscale frames to:\n%s\n', frame_counter, output_img_dir);
fprintf('Next step: Use Labelme or CVAT to create binary masks in:\n%s\n', output_mask_dir);
