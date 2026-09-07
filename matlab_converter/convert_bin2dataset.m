%% =====================================================================
%% convert_bin2dataset.m
%% MASTER PREPROCESSOR (METODE MATLAB ASLI GILANG - BERSIH & NOISE-FREE)
%% =====================================================================
%% Fitur Unggulan:
%% 1. Menggunakan Rumus Pemrosesan Sinyal Asli Gilang:
%%    - a3 = abs(RF).^0.25 (Power compression 4th-root, meredam speckle noise)
%%    - a4 = imgaussfilt(a3, 1.0) (Filter Gaussian halus, bebas bintik kasar)
%% 2. Otomatis membaca dataset 'Probefix - Sebastian' (Full Array 128 Scanlines).
%% 3. Menyimpan GAMBAR SATUAN (512x219 px) di dataset/images/.
%% 4. Menyimpan GAMBAR GABUNGAN HORIZONTAL di dataset/horizontal_stitched_filmstrip/.
%% 5. Menyimpan Matriks CSV & M-Mode Gilang di dataset/stitched_gilang/.
%% =====================================================================

clc; clear; close all;

fprintf('=====================================================================\n');
fprintf('     MASTER PREPROCESSOR (METODE ASLI GILANG - BERSIH & BEBAS NOISE) \n');
fprintf('=====================================================================\n\n');

% -------------------------------------------------------------------------
% 1. PENGATURAN PATH & DIREKTORI
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
parent_dir = fileparts(project_root);
workspace_root = fileparts(parent_dir);

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
% 2. PARAMETER AKUSTIK & PEMROSESAN GILANG
% -------------------------------------------------------------------------
c_sound       = 1540;   % Kecepatan suara jaringan (m/s)
target_height = 512;    % Tinggi resolusi gambar satuan (piksel)
gauss_sigma   = 1.0;    % Gaussian smoothing alami ala Gilang
power_exp     = 0.25;   % Power compression 0.25 (Peredam noise alami)

manifest_rows = {};
manifest_header = {'Subject', 'Trial_Name', 'Source_Bin', 'Slice_Index', 'Timestamp_Sec', 'Image_Rel_Path', 'Raw_Panorama_Rel_Path', 'Gilang_CSV_Rel_Path'};

total_all_slices = 0;
total_all_panoramas = 0;

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
        
        % ---------------------------------------------------------
        % EKSTRAKSI GAMBAR BERSIH (RUMUS ASLI GILANG)
        % ---------------------------------------------------------
        fprintf('    [1/3] Mengekstrak %d slice satuan bersih (512x%d px) ...\n', num_slices, target_width);
        
        for slice_idx = 1:num_slices
            rf_cell = RF_DATA{1, slice_idx};
            if isempty(rf_cell), continue; end
            
            % Kolom tengah untuk metode Gilang
            a2_gilang(:, slice_idx) = rf_cell(:, center_col);
            
            % 1. Power Compression Asli Gilang
            a3 = abs(double(rf_cell)).^power_exp;
            a3_min = min(a3(:));
            a3_max = max(a3(:));
            a3_norm = (a3 - a3_min) / (a3_max - a3_min + eps);
            
            % 2. Filter Gaussian 1.0 (Membersihkan noise dan bintik kasar)
            a4 = imgaussfilt(a3_norm, gauss_sigma);
            
            % 3. Isotropic Scan Conversion (512 x 219 px)
            bmode_iso = imresize(a4, [target_height, target_width], 'bicubic');
            bmode_uint8 = uint8(bmode_iso * 255);
            
            % Simpan raw untuk filmstrip
            col_start = (slice_idx - 1) * num_lines + 1;
            col_end   = slice_idx * num_lines;
            raw_horizontal(:, col_start:col_end) = uint8(a4 * 255);
            
            img_filename = sprintf('frame_%04d.png', slice_idx);
            imwrite(bmode_uint8, fullfile(trial_img_dir, img_filename));
            
            rel_img_path = fullfile('images', subject_name, trial_tag, img_filename);
            timestamp_sec = (slice_idx - 1) / fps;
            
            manifest_rows(end+1, :) = {subject_name, trial_tag, bin_files_info(f_idx).name, slice_idx, timestamp_sec, ...
                rel_img_path, ...
                fullfile('horizontal_stitched_filmstrip', subject_name, sprintf('panorama_raw_%s.png', trial_tag)), ...
                fullfile('stitched_gilang', subject_name, sprintf('%s_%s.csv', subject_name, trial_tag))};
            
            total_all_slices = total_all_slices + 1;
        end
        
        % ---------------------------------------------------------
        % SIMPAN PANORAMA FILMSTRIP & CSV
        % ---------------------------------------------------------
        pano_filename = sprintf('panorama_raw_%s.png', trial_tag);
        pano_filepath = fullfile(pano_cond_dir, pano_filename);
        imwrite(raw_horizontal, pano_filepath);
        total_all_panoramas = total_all_panoramas + 1;
        fprintf('    [2/3] Panorama Filmstrip : %s (%d x %d px)\n', pano_filename, size(raw_horizontal, 2), size(raw_horizontal, 1));
        
        csv_filename = sprintf('%s_%s.csv', subject_name, trial_tag);
        csvwrite(fullfile(gilang_cond_dir, csv_filename), a2_gilang);
        
        a3_gilang = abs(a2_gilang).^0.5;
        p_high = prctile(a3_gilang(:), 99.5);
        p_low  = prctile(a3_gilang(:), 1.0);
        a3_norm = max(0, min(1, (a3_gilang - p_low) / (p_high - p_low + eps)));
        imwrite(uint8(a3_norm * 255), fullfile(gilang_cond_dir, sprintf('stitched_gilang_%s_%s.png', subject_name, trial_tag)));
        fprintf('    [3/3] Matriks CSV Gilang : %s tersimpan\n\n', csv_filename);
        
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
