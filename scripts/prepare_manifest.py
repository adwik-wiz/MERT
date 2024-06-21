import logging
import re
import os
import random
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union
from collections import Counter
import torch
import torchaudio
import argparse
from tqdm import tqdm

_LG = logging.getLogger(__name__)

file_data_path = Path('scripts/Selected_501_files.csv')
file_data = pd.read_csv(file_data_path)

chunk_audio_data_path = Path('scripts/extract_correct_music_speech_timings.csv')
chunk_audio_data = pd.read_csv(chunk_audio_data_path)

# all_raga = file_data['processed_raga_names'].unique()

def prepare_pre_train_parent_files(test_ratio, raga_files_dict):
    X_pre_train, X_pre_val = [],[]
    for raga in raga_files_dict:
        random.shuffle(raga_files_dict[raga])
        l = len(raga_files_dict[raga])
        test_num = round(l*test_ratio) if l*test_ratio>1 else 1
        validate_num = l - test_num
        for i in range(test_num):
            X_pre_train.append(raga_files_dict[raga][i])
        for i in range(validate_num):
            X_pre_val.append(raga_files_dict[raga][i+test_num])
            
    return X_pre_train, X_pre_val

def form_pretrain_chunk_data(X,root_dir):
    X_data = []
    
    for i in range(len(X)):
        # Find rows in chunk_audio_data where Original_file_data matches the current audio file
        matching_row = chunk_audio_data[chunk_audio_data['Original File Name'] == X[i]].index
        
        # Iterate over the matching rows and append each element in the 'music' column to X_data
        for elements in chunk_audio_data['music'][matching_row]:
            for element in elements.split(','):
                element = root_dir+'/'+element
                X_data.append(element)
    
    return X_data

def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    valid_percent: float = 0.01,
    seed: int = 1317,
    extension: str = "flac",
    num_files_per_raga = 10 
) -> None:
    """Create file lists for training and validation.
    Args:
        root_dir (str or Path): The directory of the dataset.
        out_dir (str or Path): The directory to store the file lists.
        valid_percent (float, optional): The percentage of data for validation. (Default: 0.01)
        seed (int): The seed for randomly selecting the validation files.
        extension (str, optional): The extension of audio files. (Default: ``flac``)
        num_files_per_raga (int,optional) : The number of minimum files you want per raga (Default: 10)
    Returns:
        None
    """

    raga_counts = Counter(file_data['processed_raga_names'])
    using_raga = [raga for raga, count in raga_counts.items() if count >= num_files_per_raga]

    raga_files_dict = {}

    # Iterate over each unique raga and get the corresponding original file names
    for raga in using_raga:
        # Filter rows where 'processed_raga_names' matches the current raga
        raga_files = file_data[file_data['processed_raga_names'] == raga]['Original File Name'].tolist()
        # Store the result in the dictionary
        raga_files_dict[raga] = raga_files
    
    X_pre_train_parent,X_pre_validate_parent = prepare_pre_train_parent_files(valid_percent, raga_files_dict)
    X_pre_train_chunk = form_pretrain_chunk_data(X_pre_train_parent,root_dir)
    X_pre_validate_chunk = form_pretrain_chunk_data(X_pre_validate_parent,root_dir)

    assert valid_percent >= 0 and valid_percent <= 1.0

    torch.manual_seed(seed)
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    valid_f = open(out_dir / f"valid.tsv", "w") if valid_percent > 0 else None
    # search_pattern = '*.train_.*.$' # prepare for extra pattern
    with open(out_dir / f"train.tsv", "w") as train_f:
        print(root_dir, file=train_f)

        if valid_f is not None:
            print(root_dir, file=valid_f)

        for fname in tqdm(root_dir.glob(f"**/*.{extension}")):

            if args.target_rate <= 0:
                try:
                    frames = torchaudio.info(fname).num_frames
                    if(frames != 48000):      #to not include any audio with time duration not equal to 30 seconds
                        continue
                    dest = train_f if torch.rand(1) > valid_percent else valid_f
                    print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
                except:
                    _LG.warning(f"Failed to read {fname}")
            else:
                # check the original sample rate, if not equal to target rate, convert it with torchaudio,
                # and save the converted file to the converted-root-dir in the same relative path
                try:
                    sr = torchaudio.info(fname).sample_rate
                    if sr != args.target_rate:
                        wav, sr = torchaudio.load(fname)
                        wav = torchaudio.functional.resample(wav, sr, args.target_rate)
                        if fname in X_pre_train_chunk:        # to distribute the chunks to valid and train 
                            dest = train_f
                        elif fname in X_pre_validate_chunk:
                            dest = valid_f
                        else:
                            continue    
                        #dest = train_f if torch.rand(1) > valid_percent else valid_f
                        # save the converted file to the converted-root-dir in the same relative path
                        converted_fname = Path(os.path.join(args.converted_root_dir, fname.relative_to(root_dir)))
                        os.makedirs(os.path.dirname(converted_fname), exist_ok=True)
                        torchaudio.save(converted_fname, wav, args.target_rate)
                        frames = wav.shape[1]  # = torchaudio.info(converted_fname).num_frames
                        print(f"{converted_fname.relative_to(args.converted_root_dir)}\t{frames}", file=dest)
                    else:
                        if fname in X_pre_train_chunk:
                            dest = train_f
                        elif fname in X_pre_validate_chunk:
                            dest = valid_f
                        else:
                            continue  
                        frames = torchaudio.info(fname).num_frames
                        dest = train_f if torch.rand(1) > valid_percent else valid_f
                        print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
                except:
                    _LG.warning(f"Failed to read {fname}")

            file_count += 1  # Increment file counter

    if valid_f is not None:
        valid_f.close()
    _LG.info("Finished creating the file lists successfully")

if __name__ == "__main__":

    # read arguments with argparse
    parser = argparse.ArgumentParser(description="Prepare manifest files for training")
    parser.add_argument("--root-dir", type=str, default="data/audio_folder", help="root dir of the audio files, must use absolute path")
    parser.add_argument("--converted-root-dir", type=str, default="data/audio_folder_converted", help="root dir of the new audio files folder, must use absolute path")
    parser.add_argument("--target-rate", type=int, default=-1, help="")
    parser.add_argument("--out-dir", type=str, default="data/audio_manifest")
    parser.add_argument("--valid-percent", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1317)
    parser.add_argument("--extension", type=str, default="flac")
    parser.add_argument("--num-files-per-raga", type = int, default = 10, help = "number of minimum parent files you want per raga")  
    args = parser.parse_args()

    create_tsv(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        valid_percent=args.valid_percent,
        seed=args.seed,
        extension=args.extension,
        file_limit=args.file_limit  # Pass the file limit argument
    )
import logging
import re
import os
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torchaudio

import argparse
from tqdm import tqdm

_LG = logging.getLogger(__name__)

def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    valid_percent: float = 0.01,
    seed: int = 1317,
    extension: str = "flac",
    file_limit: int = 50  # Add a parameter for file limit
) -> None:
    """Create file lists for training and validation.
    Args:
        root_dir (str or Path): The directory of the dataset.
        out_dir (str or Path): The directory to store the file lists.
        valid_percent (float, optional): The percentage of data for validation. (Default: 0.01)
        seed (int): The seed for randomly selecting the validation files.
        extension (str, optional): The extension of audio files. (Default: ``flac``)
        file_limit (int, optional): The number of files to process. (Default: 1000)
    Returns:
        None
    """
    assert valid_percent >= 0 and valid_percent <= 1.0

    torch.manual_seed(seed)
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    valid_f = open(out_dir / f"valid.tsv", "w") if valid_percent > 0 else None
    # search_pattern = '*.train_.*.$' # prepare for extra pattern
    with open(out_dir / f"train.tsv", "w") as train_f:
        print(root_dir, file=train_f)

        if valid_f is not None:
            print(root_dir, file=valid_f)

        file_count = 0  # Initialize file counter

        for fname in tqdm(root_dir.glob(f"**/*.{extension}")):
            if file_count >= file_limit:
                break  # Stop processing if file limit is reached

            if args.target_rate <= 0:
                try:
                    frames = torchaudio.info(fname).num_frames
                    dest = train_f if torch.rand(1) > valid_percent else valid_f
                    print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
                except:
                    _LG.warning(f"Failed to read {fname}")
            else:
                # check the original sample rate, if not equal to target rate, convert it with torchaudio,
                # and save the converted file to the converted-root-dir in the same relative path
                try:
                    sr = torchaudio.info(fname).sample_rate
                    if sr != args.target_rate:
                        wav, sr = torchaudio.load(fname)
                        wav = torchaudio.functional.resample(wav, sr, args.target_rate)
                        dest = train_f if torch.rand(1) > valid_percent else valid_f
                        # save the converted file to the converted-root-dir in the same relative path
                        converted_fname = Path(os.path.join(args.converted_root_dir, fname.relative_to(root_dir)))
                        os.makedirs(os.path.dirname(converted_fname), exist_ok=True)
                        torchaudio.save(converted_fname, wav, args.target_rate)
                        frames = wav.shape[1]  # = torchaudio.info(converted_fname).num_frames
                        print(f"{converted_fname.relative_to(args.converted_root_dir)}\t{frames}", file=dest)
                    else:
                        frames = torchaudio.info(fname).num_frames
                        dest = train_f if torch.rand(1) > valid_percent else valid_f
                        print(f"{fname.relative_to(root_dir)}\t{frames}", file=dest)
                except:
                    _LG.warning(f"Failed to read {fname}")

            file_count += 1  # Increment file counter

    if valid_f is not None:
        valid_f.close()
    _LG.info("Finished creating the file lists successfully")

if __name__ == "__main__":

    # read arguments with argparse
    parser = argparse.ArgumentParser(description="Prepare manifest files for training")
    parser.add_argument("--root-dir", type=str, default="data/audio_folder", help="root dir of the audio files, must use absolute path")
    parser.add_argument("--converted-root-dir", type=str, default="data/audio_folder_converted", help="root dir of the new audio files folder, must use absolute path")
    parser.add_argument("--target-rate", type=int, default=-1, help="")
    parser.add_argument("--out-dir", type=str, default="data/audio_manifest")
    parser.add_argument("--valid-percent", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1317)
    parser.add_argument("--extension", type=str, default="flac")
    parser.add_argument("--file-limit", type=int, default=1000, help="number of files to process")  # Add argument for file limit
    args = parser.parse_args()

    create_tsv(
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        valid_percent=args.valid_percent,
        seed=args.seed,
        extension=args.extension,
        file_limit=args.file_limit  # Pass the file limit argument
    )
