%% =====================================================================
%% convert_bin2dataset.m
%% ULTRASOUND MASTER PREPROCESSOR (MATLAB HIGH-DEFINITION PIPELINE)
%% =====================================================================
%% Fitur Unggulan:
%% 1. Parameter Tuning HD (Sharpening, CLAHE, Noise Rejection, Gamma S-Curve)
%%    yang mereplikasi ketajaman kontras diagnostik EchoWave.
%% 2. Mengekstrak GAMBAR SATU PER SATU (B-Mode Isotropik 2D HD)
%%    ke dalam dataset/images/ untuk anotasi AI (Labelme/CVAT).
%% 3. Menyusun GAMBAR GABUNGAN HORIZONTAL (Raw 33-Scanline Filmstrip)
%%    ke dalam dataset/horizontal_stitched_filmstrip/ langsung setelah slice selesai.
%% 4. Mengekspor Matriks Numerik .CSV & M-Mode Metode Gilang.
%% =====================================================================

clc; clear; close all;

fprintf('=====================================================================\n');
fprintf('     ULTRASOUND MASTER PREPROCESSOR (HD PARAMETER PIPELINE)          \n');
fprintf('=====================================================================\n\n');

% -------------------------------------------------------------------------
% 1. PENGATURAN PATH & DIREKTORI
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
parent_dir = fileparts(project_root);

britty_dir           = fullfile(parent_dir, 'britty');
output_img_base      = fullfile(parent_dir, 'dataset', 'images');
output_mask_base     = fullfile(parent_dir, 'dataset', 'masks');
output_panorama_base = fullfile(parent_dir, 'dataset', 'horizontal_stitched_filmstrip');
output_gilang_base   = fullfile(parent_dir, 'dataset', 'stitched_gilang');

if ~exist(output_img_base, 'dir'), mkdir(output_img_base); end
if ~exist(output_mask_base, 'dir'), mkdir(output_mask_base); end
if ~exist(output_panorama_base, 'dir'), mkdir(output_panorama_base); end
if ~exist(output_gilang_base, 'dir'), mkdir(output_gilang_base); end

addpath(script_dir);

% -------------------------------------------------------------------------
% 2. PARAMETER TUNING HIGH-DEFINITION (DAPAT DIATUR VIA IMAGE LAB)
% -------------------------------------------------------------------------
dr_dB         = 78.0;   % Dynamic Range (30 - 90 dB)
gain_val      = 1.20;   % Gain Multiplier (0.5 - 3.0)
rejection_pct = 8.0;    % Noise Rejection Clip % (0 - 25% memotong noise dasar menjadi hitam pekat)
gamma_val     = 1.10;   % Gamma / Contrast S-Curve (0.5 - 2.0)
sharpen_val   = 1.50;   % Unsharp Masking Texture Enhancement (0.0 - 4.0 untuk ketajaman fasikel)
clahe_limit   = 0.02;   % CLAHE Adaptive Contrast Limit (0.00 - 0.05)
gauss_sigma   = 0.00;   % Gaussian Smoothing (0.00 = OFF agar tidak blur)

c_sound       = 1540;   % Kecepatan rambat suara pada jaringan otot (m/s)
target_height = 512;    % Tinggi resolusi gambar satuan (piksel)
center_col    = 17;     % Kolom tengah untuk metode Gilang

conditions = {'5rm', '10rm'};
manifest_rows = {};
manifest_header = {'Condition', 'Trial_Name', 'Source_Bin', 'Slice_Index', 'Timestamp_Sec', 'Image_Rel_Path', 'Raw33_Panorama_Rel_Path', 'Gilang_CSV_Rel_Path'};

total_all_slices = 0;
total_all_panoramas = 0;

% -------------------------------------------------------------------------
% 3. LOOP PEMROSESAN SETIAP KONDISI & TRIAL
% -------------------------------------------------------------------------
for c_idx = 1:length(conditions)
    cond_name = conditions{c_idx};
    cond_path = fullfile(britty_dir, cond_name);
    
    if ~exist(cond_path, 'dir')
        fprintf('[Warning] Folder tidak ditemukan: %s\n', cond_path);
        continue;
    end
    
    pano_cond_dir   = fullfile(output_panorama_base, cond_name);
    gilang_cond_dir = fullfile(output_gilang_base, cond_name);
    if ~exist(pano_cond_dir, 'dir'), mkdir(pano_cond_dir); end
    if ~exist(gilang_cond_dir, 'dir'), mkdir(gilang_cond_dir); end
    
    sub_dirs = dir(cond_path);
    sub_dirs = sub_dirs([sub_dirs.isdir]);
    sub_dirs = sub_dirs(~ismember({sub_dirs.name}, {'.', '..'}));
    
    fprintf('=====================================================================\n');
    fprintf(' Memproses Kondisi: [%s]\n', cond_name);
    fprintf('=====================================================================\n');
    
    trial_count = 0;
    
    for s_idx = 1:length(sub_dirs)
        sub_folder_name = sub_dirs(s_idx).name;
        if ~contains(lower(sub_folder_name), 'open'), continue; end
        
        current_open_dir = fullfile(cond_path, sub_folder_name);
        bin_files = dir(fullfile(current_open_dir, '*.bin'));
        if isempty(bin_files), continue; end
        
        for f_idx = 1:length(bin_files)
            trial_count = trial_count + 1;
            file_name = bin_files(f_idx).name;
            [~, raw_name, ~] = fileparts(file_name);
            
            trial_tag = sprintf('trial_%02d_%s_%s', trial_count, sub_folder_name, raw_name);
            trial_tag = regexprep(trial_tag, '[^a-zA-Z0-9_-]', '_');
            
            trial_img_dir  = fullfile(output_img_base, cond_name, trial_tag);
            trial_mask_dir = fullfile(output_mask_base, cond_name, trial_tag);
            if ~exist(trial_img_dir, 'dir'), mkdir(trial_img_dir); end
            if ~exist(trial_mask_dir, 'dir'), mkdir(trial_mask_dir); end
            
            fprintf('\n--> [Take %02d] File: %s/%s/%s\n', trial_count, cond_name, sub_folder_name, file_name);
            
            try
                [RF_DATA, RF_DATA_Q, HEADER, transducer_code, flag] = import_RFdata2MATLAB([current_open_dir, filesep], file_name);
                if flag == 0 || isempty(RF_DATA)
                    fprintf('    [Warning] File tidak dapat dibaca: %s\n', file_name);
                    continue;
                end
                
                hdr = HEADER{1};
                num_samples = hdr.Length_of_RF_row; % 2959
                num_lines   = hdr.Number_of_RF_rows; % 33
                num_slices  = size(RF_DATA, 2);
                fps         = hdr.frame_rate;
                if fps <= 0, fps = 120.0; end
                
                % Skala fisik & Aspek Rasio
                dz_mm = (c_sound * (hdr.Sampling_period_ns * 1e-9) / 2) * 1000;
                total_depth_mm = num_samples * dz_mm;
                if isfield(hdr, 'beam_x') && length(hdr.beam_x) > 1
                    total_width_mm = (max(hdr.beam_x) - min(hdr.beam_x)) * 10;
                else
                    total_width_mm = 9.9;
                end
                phys_aspect_ratio = total_width_mm / total_depth_mm;
                target_width = round(target_height * phys_aspect_ratio); % 89 px
                mm_per_pixel = total_depth_mm / target_height;
                
                fprintf('    Info: %d frame | Kedalaman: %.1f mm | Lebar: %.1f mm | %.1f FPS\n', ...
                    num_slices, total_depth_mm, total_width_mm, fps);
                
                raw33_horizontal = zeros(num_samples, num_lines * num_slices, 'uint8');
                a2_gilang = zeros(num_samples, num_slices);
                
                rej_norm = rejection_pct / 100.0;
                
                % ---------------------------------------------------------
                % TAHAP 1: EKSTRAKSI GAMBAR SATUAN HIGH-DEFINITION
                % ---------------------------------------------------------
                fprintf('    [1/3] Mengekstrak %d slice HD (512x%d px) ...\n', num_slices, target_width);
                
                for slice_idx = 1:num_slices
                    rf_cell = RF_DATA{1, slice_idx};
                    if isempty(rf_cell), continue; end
                    
                    a2_gilang(:, slice_idx) = rf_cell(:, min(center_col, size(rf_cell, 2)));
                    
                    % 1. Demodulasi Envelope & Gain
                    env = abs(hilbert(double(rf_cell)));
                    env_max = max(env(:));
                    if env_max == 0, env_max = 1; end
                    env_norm = (env / env_max) * gain_val;
                    
                    % 2. Dynamic Range Compression (DR in dB)
                    env_dB = 20 * log10(env_norm + eps);
                    bmode = (env_dB + dr_dB) / dr_dB;
                    bmode = max(0, min(1, bmode));
                    
                    % 3. Noise Rejection (Potong noise latar belakang menjadi hitam pekat)
                    if rej_norm > 0
                        bmode(bmode < rej_norm) = 0;
                        bmode = (bmode - rej_norm) / (1 - rej_norm + eps);
                        bmode = max(0, min(1, bmode));
                    end
                    
                    % 4. Gamma Contrast Curve
                    bmode = bmode.^gamma_val;
                    
                    % Simpan ke matriks gabungan horizontal 33-scanline
                    bmode_raw_uint8 = uint8(bmode * 255);
                    col_start = (slice_idx - 1) * num_lines + 1;
                    col_end   = slice_idx * num_lines;
                    raw33_horizontal(:, col_start:col_end) = bmode_raw_uint8;
                    
                    % 5. Gaussian Filter (hanya jika diaktifkan > 0)
                    if gauss_sigma > 0.05
                        bmode = imgaussfilt(bmode, gauss_sigma);
                    end
                    
                    % 6. Isotropic Scan Conversion
                    bmode_iso = imresize(bmode, [target_height, target_width], 'bicubic');
                    
                    % 7. Unsharp Masking (Penajaman tekstur fasikel & garis fascia putih)
                    if sharpen_val > 0
                        bmode_iso = imsharpen(bmode_iso, 'Radius', 1.5, 'Amount', sharpen_val);
                        bmode_iso = max(0, min(1, bmode_iso));
                    end
                    
                    % 8. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
                    if clahe_limit > 0.001
                        bmode_iso = adapthisteq(bmode_iso, 'ClipLimit', clahe_limit, 'NumTiles', [8 8]);
                    end
                    
                    % Simpan gambar satuan
                    img_filename = sprintf('frame_%04d.png', slice_idx);
                    imwrite(uint8(bmode_iso * 255), fullfile(trial_img_dir, img_filename));
                    
                    rel_img_path = fullfile('images', cond_name, trial_tag, img_filename);
                    timestamp_sec = (slice_idx - 1) / fps;
                    
                    manifest_rows(end+1, :) = {cond_name, trial_tag, file_name, slice_idx, timestamp_sec, ...
                        rel_img_path, ...
                        fullfile('horizontal_stitched_filmstrip', cond_name, sprintf('panorama_raw33_%s.png', trial_tag)), ...
                        fullfile('stitched_gilang', cond_name, sprintf('%s_%s.csv', cond_name, raw_name))};
                    
                    total_all_slices = total_all_slices + 1;
                end
                
                % ---------------------------------------------------------
                % TAHAP 2: SIMPAN GAMBAR GABUNGAN HORIZONTAL (MERGER PANORAMA)
                % ---------------------------------------------------------
                pano_filename = sprintf('panorama_raw33_%s.png', trial_tag);
                pano_filepath = fullfile(pano_cond_dir, pano_filename);
                imwrite(raw33_horizontal, pano_filepath);
                total_all_panoramas = total_all_panoramas + 1;
                
                fprintf('    [2/3] Menyimpan Gambar Gabungan: %s (%d x %d px)\n', ...
                    pano_filename, size(raw33_horizontal, 2), size(raw33_horizontal, 1));
                
                % ---------------------------------------------------------
                % TAHAP 3: SIMPAN DATA STITCHED & CSV METODE GILANG
                % ---------------------------------------------------------
                csv_filename = sprintf('%s_%s.csv', cond_name, raw_name);
                csvwrite(fullfile(gilang_cond_dir, csv_filename), a2_gilang);
                
                a3_gilang = abs(a2_gilang).^0.5;
                p_high = prctile(a3_gilang(:), 99.5);
                p_low  = prctile(a3_gilang(:), 1.0);
                a3_norm = max(0, min(1, (a3_gilang - p_low) / (p_high - p_low + eps)));
                imwrite(uint8(a3_norm * 255), fullfile(gilang_cond_dir, sprintf('stitched_gilang_%s_%s.png', cond_name, raw_name)));
                
                fprintf('    [3/3] Menyimpan Matriks CSV Gilang: %s\n', csv_filename);
                
            catch ME
                fprintf('    [Error] Gagal memproses %s: %s\n', file_name, ME.message);
            end
        end
    end
end

% -------------------------------------------------------------------------
% 4. SIMPAN MASTER MANIFEST & KALIBRASI SKALA FISIK
% -------------------------------------------------------------------------
if ~isempty(manifest_rows)
    manifest_table = cell2table(manifest_rows, 'VariableNames', manifest_header);
    manifest_path = fullfile(parent_dir, 'dataset', 'dataset_manifest.csv');
    writetable(manifest_table, manifest_path);
    fprintf('\nManifest tersimpan di: %s\n', manifest_path);
end

scale_table = table(total_depth_mm, total_width_mm, target_height, target_width, mm_per_pixel, ...
    'VariableNames', {'Depth_mm', 'Width_mm', 'Height_px', 'Width_px', 'MM_per_pixel'});
writetable(scale_table, fullfile(parent_dir, 'dataset', 'spatial_calibration.csv'));

fprintf('\n=====================================================================\n');
fprintf('                 SELURUH PREPROCESSING HD SELESAI!                   \n');
fprintf('=====================================================================\n');
fprintf('1. Gambar Satuan HD (Untuk Labeling) : %d file di dataset/images/\n', total_all_slices);
fprintf('2. Gambar Gabungan (Merger Panorama) : %d file di dataset/horizontal_stitched_filmstrip/\n', total_all_panoramas);
fprintf('3. Data CSV & M-Mode Gilang          : %d file di dataset/stitched_gilang/\n', total_all_panoramas);
fprintf('4. Skala Fisik Citra                 : 1 piksel = %.4f mm\n', mm_per_pixel);
fprintf('=====================================================================\n');
