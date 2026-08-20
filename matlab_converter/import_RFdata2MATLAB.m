function [RF_DATA, RF_DATA_Q, HEADER, transducer_code, flag] = import_RFdata2MATLAB(DIR, filename)
%% Function which allows to import RF data and other data acquisition parameters needed for imaging 
%%
%% IN:
%% filename - *.bin binary file recorded using ArtUS device and artus_rf_test sample
%% DIR - directory of the file
%%
%% OUT:
%% RF_DATA - cell format (1 x K - number of RF frames and inside N (number of RF samples) X M (number of RF rows)) 
%% RF_DATA_Q - Q component obtained after Hilbert transform
%% HEADER - RF data acquisition parameters
%% transducer_code - probe type / code
%% flag - indicates successful file reading (1 = success, 0 = failed)

% Initialize all output arguments to avoid unassigned output errors
RF_DATA = {};
RF_DATA_Q = {};
HEADER = {};
transducer_code = 'Unknown';
flag = 0;

% Ensure DIR has trailing path separator
if ~isempty(DIR) && DIR(end) ~= '/' && DIR(end) ~= '\'
    DIR = [DIR, filesep];
end

full_file_path = [DIR, filename];

fid = fopen(full_file_path, 'r');
if (fid == -1)
    warning('Failed to open file: %s', full_file_path);
    return;
end

% Get file size in bytes
fseek(fid, 0, 'eof');
File_size = ftell(fid);
fseek(fid, 0, 'bof');

if (File_size == 0)
    fclose(fid);
    warning('Selected file is empty: %s', full_file_path);
    return;
end

% Extract transducer code safely
if length(filename) >= 25
    transducer_code = filename(21:end-4);
else
    transducer_code = 'Generic';
end

% Read header identifier
A = fread(fid, 6, '*char')';
A_temp = A;

if strcmp(A, 'RF0001') || strcmp(A, 'RF0002') || strcmp(A, 'RF0003') || strcmp(A, 'RF0004')
    number_of_header_els = 11;
else
    number_of_header_els = 7;
    fseek(fid, 0, 'bof'); % Reset position to beginning
end

size_in_bytes_global = 0;
stop = true;
i = 1;

while (stop)
    if (number_of_header_els == 11)
        A = fread(fid, number_of_header_els, 'int32');
        if length(A) < 11
            break;
        end
        header.number_of_frames = A(1);
        header.header_size = A(2);
        header.frame_size = A(3);
        header.source_ID = A(4);
        header.tx_frequency = A(5);
        header.frame_rate = A(6) / 100;
        header.Length_of_RF_row = A(7);
        header.Number_of_RF_rows = A(8);
        header.Sampling_period_ns = A(9);
        header.sample_size = A(10);
        start_depth = A(11);
    else
        A = fread(fid, number_of_header_els, 'int32');
        if length(A) < 7
            break;
        end
        header.tx_frequency = A(1);
        header.frame_rate = A(2) / 100;
        header.Length_of_RF_row = A(3);
        header.Number_of_RF_rows = A(4);
        header.Sampling_period_ns = A(5);
        header.sample_size = A(6);
        start_depth = A(7);
        header.source_ID = 0;
    end

    if (header.sample_size ~= 16) && (header.sample_size ~= 32)
        warning('Wrong file structure in %s (sample_size = %d)', filename, header.sample_size);
        fclose(fid);
        return;
    end

    Start_point_position_and_orient = fread(fid, 3 * header.Number_of_RF_rows, 'int32');
    if length(Start_point_position_and_orient) < (3 * header.Number_of_RF_rows)
        break;
    end

    header.angle = Start_point_position_and_orient(3:3:end) ./ 1000000;
    header.beam_x = Start_point_position_and_orient(1:3:end) / 10000 + (start_depth / 10 * sin(header.angle));
    header.beam_y = Start_point_position_and_orient(2:3:end) / 10000 + (start_depth / 10 * cos(header.angle));
    header.Speed_of_sound = 1540;

    if strcmp(A_temp, 'RF0003') || strcmp(A_temp, 'RF0004')
        header.time_stamps = fread(fid, header.Number_of_RF_rows, 'uint32');
    end

    HEADER{i} = header;

    if (strcmp(A_temp, 'RF0002') || strcmp(A_temp, 'RF0003') || strcmp(A_temp, 'RF0004')) && (header.source_ID == 4)
        size_of_frame = header.Length_of_RF_row * header.Number_of_RF_rows * 2;
        frame_to_reshape = fread(fid, size_of_frame, 'int16');
        if length(frame_to_reshape) < size_of_frame
            break;
        end
        I_component = frame_to_reshape(1:end/2);
        Q_component = frame_to_reshape(end/2+1:end);
        RF_DATA{i} = reshape(I_component, header.Length_of_RF_row, header.Number_of_RF_rows);
        RF_DATA_Q{i} = reshape(Q_component, header.Length_of_RF_row, header.Number_of_RF_rows);
    else
        size_of_frame = header.Length_of_RF_row * header.Number_of_RF_rows;
        frame_to_reshape = fread(fid, size_of_frame, 'int16');
        if length(frame_to_reshape) < size_of_frame
            break;
        end
        RF_DATA{i} = reshape(frame_to_reshape, header.Length_of_RF_row, header.Number_of_RF_rows);
        RF_DATA_Q{i} = zeros(header.Length_of_RF_row, header.Number_of_RF_rows);
    end

    if strcmp(A_temp, 'RF0003') || strcmp(A_temp, 'RF0004')
        size_in_bytes_run = (number_of_header_els * 4) + (3 * header.Number_of_RF_rows * 4) + (size_of_frame * 2) + (header.Number_of_RF_rows * 4);
    else
        size_in_bytes_run = (number_of_header_els * 4) + (3 * header.Number_of_RF_rows * 4) + (size_of_frame * 2);
    end

    size_in_bytes_global = size_in_bytes_global + size_in_bytes_run;
    i = i + 1;
    if (File_size - size_in_bytes_global) >= size_in_bytes_run
        stop = true;
    else
        stop = false;
    end
end

fclose(fid);
flag = 1;
end
