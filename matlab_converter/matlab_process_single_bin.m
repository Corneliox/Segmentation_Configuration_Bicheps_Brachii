function matlab_process_single_bin(binFilePath, outputDir)
    % MATLAB_PROCESS_SINGLE_BIN
    % Memproses 1 file .bin ArtUs menjadi frame-frame PNG bersih dengan metode Gilang
    fprintf('=== MATLAB BICEPS BRACHII RF PROCESSOR (METODE GILANG) ===\n');
    fprintf('Input BIN : %s\n', binFilePath);
    fprintf('Output Dir: %s\n', outputDir);

    if ~exist(binFilePath, 'file')
        error('File .bin tidak ditemukan: %s', binFilePath);
    end

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    scriptDir = fileparts(mfilename('fullpath'));
    addpath(scriptDir);

    [binFolder, binName, binExt] = fileparts(binFilePath);
    binFullName = [binName, binExt];

    fprintf('Membaca RF Data via import_RFdata2MATLAB...\n');
    [RF_DATA, RF_DATA_Q, HEADER, transducer_code, flag] = import_RFdata2MATLAB([binFolder, filesep], binFullName);

    if flag == 0 || isempty(RF_DATA)
        error('Gagal membaca data dari file: %s', binFullName);
    end

    hdr = HEADER{1};
    num_samples = hdr.Length_of_RF_row;
    num_lines = hdr.Number_of_RF_rows;
    num_slices = size(RF_DATA, 2);
    fps = hdr.frame_rate;
    if fps <= 0, fps = 60.0; end

    c_sound = 1540;
    dz_mm = (c_sound * (hdr.Sampling_period_ns * 1e-9) / 2) * 1000;
    total_depth_mm = num_samples * dz_mm;

    if isfield(hdr, 'beam_x') && length(hdr.beam_x) > 1
        total_width_mm = (max(hdr.beam_x) - min(hdr.beam_x)) * 10;
    else
        total_width_mm = 38.4;
    end

    target_height = 512;
    aspect_ratio = total_width_mm / total_depth_mm;
    target_width = round(target_height * aspect_ratio);

    power_exp = 0.25;
    gauss_sigma = 1.0;

    fprintf('Parameter: %d slices | %.1f FPS | Ukuran Output: %d x %d px\n', ...
        num_slices, fps, target_width, target_height);

    for i = 1:num_slices
        rf_cell = RF_DATA{1, i};
        if isempty(rf_cell), continue; end

        % 1. Power compression Gilang 0.25
        a3 = abs(double(rf_cell)).^power_exp;
        a3_min = min(a3(:));
        a3_max = max(a3(:));
        a3_norm = (a3 - a3_min) / (a3_max - a3_min + eps);

        % 2. Filter Gaussian 1.0 (Halus & Bersih)
        a4 = imgaussfilt(a3_norm, gauss_sigma);

        % 3. Isotropic Resize Bicubic
        bmode_iso = imresize(a4, [target_height, target_width], 'bicubic');
        bmode_uint8 = uint8(bmode_iso * 255);

        outPath = fullfile(outputDir, sprintf('frame_%04d.png', i));
        imwrite(bmode_uint8, outPath);
    end

    % Simpan metadata JSON / MAT
    metaFile = fullfile(outputDir, 'metadata.txt');
    fid = fopen(metaFile, 'w');
    if fid ~= -1
        fprintf(fid, 'Source_Bin: %s\n', binFullName);
        fprintf(fid, 'Num_Frames: %d\n', num_slices);
        fprintf(fid, 'FPS: %.2f\n', fps);
        fprintf(fid, 'Depth_mm: %.2f\n', total_depth_mm);
        fprintf(fid, 'Width_mm: %.2f\n', total_width_mm);
        fprintf(fid, 'Pixel_Spacing_mm: %.4f\n', total_depth_mm / target_height);
        fprintf(fid, 'Width_px: %d\n', target_width);
        fprintf(fid, 'Height_px: %d\n', target_height);
        fclose(fid);
    end

    fprintf('✓ Sukses mengekstrak %d frames ke: %s\n', num_slices, outputDir);
end
