%% =====================================================================
%% convert_bin2dataset.m
%% ULTRASOUND MASTER PREPROCESSOR (PROBEFIX - SEBASTIAN PIPELINE)
%% =====================================================================
%% Fitur Unggulan:
%% 1. Otomatis membaca dataset 'Probefix - Sebastian' (Full Array 128 Scanlines, L12-5N40-A4).
%% 2. Mengekstrak GAMBAR SATU PER SATU (B-Mode Isotropik 2D HD 512x219 px)
%%    ke dalam dataset/images/ untuk anotasi AI (Labelme/CVAT).
%% 3. Menyusun GAMBAR GABUNGAN HORIZONTAL (Raw 128-Scanline Filmstrip)
%%    ke dalam dataset/horizontal_stitched_filmstrip/ langsung setelah slice selesai.
%% 4. Mengekspor Matriks Numerik .CSV & M-Mode Metode Gilang.
%% 5. Menyimpan Master Manifest & Kalibrasi Skala Fisik (mm/pixel).
%% =====================================================================

clc; clear; close all;

fprintf('=====================================================================\n');
fprintf('     ULTRASOUND MASTER PREPROCESSOR (PROBEFIX - SEBASTIAN PIPELINE)  \n');
fprintf('=====================================================================\n\n');

% -------------------------------------------------------------------------
% 1. PENGATURAN PATH & DIREKTORI
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
parent_dir = fileparts(project_root);
workspace_root = fileparts(parent_dir);

% Deteksi folder 'Probefix - Sebastian'
possible_sebastian = { ...
    fullfile(workspace_root, 'Probefix - Sebastian'), ...
    fullfile(parent_dir, 'Probefix - Sebastian'), ...
    'D:\~Ideas n Innovation\~~Taiwan\AU\Probefix Segmentation\Probefix - Sebastian' ...
};

sebastian_dir = '';
for i = 1:length(possible_sebastian)
    if exist(possible_sebastian{i}, 'dir')
        sebastian_dir = possible_sebastian{i};
        break;
    end
end

if isempty(sebastian_dir)
    error('Folder "Probefix - Sebastian" tidak ditemukan!');
end

fprintf('Direktori Input  : %s\n', sebastian_dir);
output_base = fullfile(parent_dir, 'dataset');
output_img_base      = fullfile(output_base, 'images');
output_mask_base     = fullfile(output_base, 'masks');
output_panorama_base = fullfile(output_base, 'horizontal_stitched_filmstrip');
output_gilang_base   = fullfile(output_base, 'stitched_gilang');

if ~exist(output_img_base, 'dir'), mkdir(output_img_base); end
if ~exist(output_mask_base, 'dir'), mkdir(output_mask_base); end
if ~exist(output_panorama_base, 'dir'), mkdir(output_panorama_base); end
if ~exist(output_gilang_base, 'dir'), mkdir(output_gilang_base); end

addpath(script_dir);

% -------------------------------------------------------------------------
% 2. PARAMETER TUNING HIGH-DEFINITION
% -------------------------------------------------------------------------
dr_dB         = 78.0;   % Dynamic Range (30 - 90 dB)
gain_val      = 1.20;   % Gain Multiplier (0.5 - 3.0)
rejection_pct = 8.0;    % Noise Rejection Clip % (0 - 25%)
gamma_val     = 1.10;   % Gamma / Contrast S-Curve (0.5 - 2.0)
sharpen_val   = 1.50;   % Unsharp Masking Texture Enhancement (0.0 - 4.0)
clahe_limit   = 0.02;   % CLAHE Adaptive Contrast Limit (0.00 - 0.05)
gauss_sigma   = 0.00;   % Gaussian Smoothing (0.00 = OFF)

c_sound       = 1540;   % Kecepatan rambat suara pada jaringan otot (m/s)
target_height = 512;    % Tinggi resolusi gambar satuan (piksel)

manifest_rows = {};
manifest_header = {'Subject', 'Trial_Name', 'Source_Bin', 'Slice_Index', 'Timestamp_Sec', 'Image_Rel_Path', 'Raw_Panorama_Rel_Path', 'Gilang_CSV_Rel_Path'};

total_all_slices = 0;
total_all_panoramas = 0;

% Cari seluruh file .bin secara rekursif
bin_files_info = dir(fullfile(sebastian_dir, '**', '*.bin'));
if isempty(bin_files_info)
    error('Tidak ditemukan file .bin di dalam %s', sebastian_dir);
end

fprintf('Ditemukan %d file .bin untuk diproses.\n\n', length(bin_files_info));

% -------------------------------------------------------------------------
% 3. LOOP PEMROSESAN SETIAP FILE .BIN
% -------------------------------------------------------------------------
for f_idx = 1:length(bin_files_info)
    bin_file_path = fullfile(bin_files_info(f_idx).folder, bin_files_info(f_idx).name);
    
    % Ekstrak nama subject dari nama subfolder
    rel_folder = strrep(bin_files_info(f_idx).folder, sebastian_dir, '');
    rel_folder = regexprep(rel_folder, '^[\\/]+', '');
    if isempty(rel_folder)
        subject_name = 'Sebastian';
    else
        subject_name = regexprep(rel_folder, '[^a-zA-Z0-9_-]', '_');
    end
    
    [~, raw_name, ~] = fileparts(bin_files_info(f_idx).name);
    raw_name_clean = regexprep(raw_name, '[^a-zA-Z0-9_-]', '_');
    trial_tag = sprintf('trial_%s', raw_name_clean);
    
    trial_img_dir  = fullfile(output_img_base, subject_name, trial_tag);
    trial_mask_dir = fullfile(output_mask_base, subject_name, trial_tag);
    pano_cond_dir  = fullfile(output_panorama_base, subject_name);
    gilang_cond_dir = fullfile(output_gilang_base, subject_name);
    
    if ~exist(trial_img_dir, 'dir'), mkdir(trial_img_dir); end
    if ~exist(trial_mask_dir, 'dir'), mkdir(trial_mask_dir); end
    if ~exist(pano_cond_dir, 'dir'), mkdir(pano_cond_dir); end
    if ~exist(gilang_cond_dir, 'dir'), mkdir(gilang_cond_dir); end
    
    fprintf('=====================================================================\n');
    fprintf(' [%d/%d] Memproses Subject: %s | File: %s\n', f_idx, length(bin_files_info), subject_name, bin_files_info(f_idx).name);
    fprintf('=====================================================================\n');
    
    try
        [RF_DATA, RF_DATA_Q, HEADER, transducer_code, flag] = import_RFdata2MATLAB([bin_files_info(f_idx).folder, filesep], bin_files_info(f_idx).name);
        if flag == 0 || isempty(RF_DATA)
            fprintf('    [Warning] File tidak dapat dibaca: %s\n', bin_files_info(f_idx).name);
            continue;
        end
        
        hdr = HEADER{1};
        num_samples = hdr.Length_of_RF_row; % e.g. 4673
        num_lines   = hdr.Number_of_RF_rows; % e.g. 128
        num_slices  = size(RF_DATA, 2);
        fps         = hdr.frame_rate;
        if fps <= 0, fps = 120.0; end
        
        % Skala fisik & Aspek Rasio
        dz_mm = (c_sound * (hdr.Sampling_period_ns * 1e-9) / 2) * 1000;
        total_depth_mm = num_samples * dz_mm;
        if isfield(hdr, 'beam_x') && length(hdr.beam_x) > 1
            total_width_mm = (max(hdr.beam_x) - min(hdr.beam_x)) * 10;
        else
            total_width_mm = 38.4;
        end
        phys_aspect_ratio = total_width_mm / total_depth_mm;
        target_width = round(target_height * phys_aspect_ratio); % e.g. 219 px
        mm_per_pixel = total_depth_mm / target_height;
        center_col = round(num_lines / 2);
        
        fprintf('    Info: %d frame | Kedalaman: %.1f mm | Lebar: %.1f mm | Aspect Ratio: %.3f | %.1f FPS\n', ...
            num_slices, total_depth_mm, total_width_mm, phys_aspect_ratio, fps);
        
        raw_horizontal = zeros(num_samples, num_lines * num_slices, 'uint8');
        a2_gilang = zeros(num_samples, num_slices);
        rej_norm = rejection_pct / 100.0;
        
        % ---------------------------------------------------------
        % TAHAP 1: EKSTRAKSI GAMBAR SATUAN HIGH-DEFINITION
        % ---------------------------------------------------------
        fprintf('    [1/3] Mengekstrak %d slice satuan (512x%d px) ...\n', num_slices, target_width);
        
        for slice_idx = 1:num_slices
            rf_cell = RF_DATA{1, slice_idx};
            if isempty(rf_cell), continue; end
            
            % Kolom tengah untuk metode Gilang
            a2_gilang(:, slice_idx) = rf_cell(:, center_col);
            
            % Demodulasi Envelope & Log-Compression
            env = abs(hilbert(double(rf_cell)));
            env_max = max(env(:));
            if env_max == 0, env_max = 1; end
            env_norm = (env / env_max) * gain_val;
            
            env_dB = 20 * log10(env_norm + eps);
            bmode = (env_dB + dr_dB) / dr_dB;
            bmode = max(0, min(1, bmode));
            
            % Noise Rejection
            if rej_norm > 0
                bmode(bmode < rej_norm) = 0;
                bmode = (bmode - rej_norm) / (1 - rej_norm + eps);
                bmode = max(0, min(1, bmode));
            end
            
            bmode = bmode.^gamma_val;
            
            % Simpan raw scanlines ke matriks gabungan horizontal
            bmode_raw_uint8 = uint8(bmode * 255);
            col_start = (slice_idx - 1) * num_lines + 1;
            col_end   = slice_idx * num_lines;
            raw_horizontal(:, col_start:col_end) = bmode_raw_uint8;
            
            % Isotropic Scaling untuk slice satuan (512x219 px)
            bmode_iso = imresize(bmode, [target_height, target_width], 'bicubic');
            
            % Unsharp Masking & CLAHE (Optional HD enhancement)
            if sharpen_val > 0
                bmode_iso = imsharpen(bmode_iso, 'Radius', 1.5, 'Amount', sharpen_val);
                bmode_iso = max(0, min(1, bmode_iso));
            end
            if clahe_limit > 0.001
                bmode_iso = adapthisteq(bmode_iso, 'ClipLimit', clahe_limit, 'NumTiles', [8 8]);
            end
            
            img_filename = sprintf('frame_%04d.png', slice_idx);
            imwrite(uint8(bmode_iso * 255), fullfile(trial_img_dir, img_filename));
            
            rel_img_path = fullfile('images', subject_name, trial_tag, img_filename);
            timestamp_sec = (slice_idx - 1) / fps;
            
            manifest_rows(end+1, :) = {subject_name, trial_tag, bin_files_info(f_idx).name, slice_idx, timestamp_sec, ...
                rel_img_path, ...
                fullfile('horizontal_stitched_filmstrip', subject_name, sprintf('panorama_raw_%s.png', trial_tag)), ...
                fullfile('stitched_gilang', subject_name, sprintf('%s_%s.csv', subject_name, trial_tag))};
            
            total_all_slices = total_all_slices + 1;
        end
        
        % ---------------------------------------------------------
        % TAHAP 2: SIMPAN GAMBAR GABUNGAN HORIZONTAL
        % ---------------------------------------------------------
        pano_filename = sprintf('panorama_raw_%s.png', trial_tag);
        pano_filepath = fullfile(pano_cond_dir, pano_filename);
        imwrite(raw_horizontal, pano_filepath);
        total_all_panoramas = total_all_panoramas + 1;
        fprintf('    [2/3] Panorama Filmstrip : %s (%d x %d px)\n', pano_filename, size(raw_horizontal, 2), size(raw_horizontal, 1));
        
        % ---------------------------------------------------------
        % TAHAP 3: SIMPAN DATA STITCHED & CSV METODE GILANG
        % ---------------------------------------------------------
        csv_filename = sprintf('%s_%s.csv', subject_name, trial_tag);
        csvwrite(fullfile(gilang_cond_dir, csv_filename), a2_gilang);
        
        a3_gilang = abs(a2_gilang).^0.5;
        p_high = prctile(a3_gilang(:), 99.5);
        p_low  = prctile(a3_gilang(:), 1.0);
        a3_norm = max(0, min(1, (a3_gilang - p_low) / (p_high - p_low + eps)));
        imwrite(uint8(a3_norm * 255), fullfile(gilang_cond_dir, sprintf('stitched_gilang_%s_%s.png', subject_name, trial_tag)));
        fprintf('    [3/3] Matriks CSV Gilang : %s tersimpan\n\n', csv_fname);
        
    catch ME
        fprintf('    [Error] Gagal memproses %s: %s\n', bin_files_info(f_idx).name, ME.message);
    end
end

% -------------------------------------------------------------------------
% 4. SIMPAN MASTER MANIFEST & KALIBRASI SKALA FISIK
% -------------------------------------------------------------------------
if ~isempty(manifest_rows)
    manifest_table = cell2table(manifest_rows, 'VariableNames', manifest_header);
    manifest_path = fullfile(output_base, 'dataset_manifest.csv');
    writetable(manifest_table, manifest_path);
    fprintf('\n[✓] Manifest tersimpan di: %s (%d total frames)\n', manifest_path, total_all_slices);
end

scale_table = table(total_depth_mm, total_width_mm, target_height, target_width, mm_per_pixel, ...
    'VariableNames', {'Depth_mm', 'Width_mm', 'Height_px', 'Width_px', 'MM_per_pixel'});
writetable(scale_table, fullfile(output_base, 'spatial_calibration.csv'));
fprintf('[✓] Kalibrasi Skala Fisik: 1 piksel = %.4f mm\n\n', mm_per_pixel);

fprintf('=====================================================================\n');
fprintf('                 SELURUH PREPROCESSING SELESAI!                      \n');
fprintf('=====================================================================\n');
