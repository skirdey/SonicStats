import os
import time
import argparse
import logging
import pandas as pd
import torchaudio
import torch
import numpy as np

from compute_metrics import compute_metrics  # Import your compute_metrics function

def main():
    parser = argparse.ArgumentParser(description="Compute PESQ, STOI, CSIG, CBAK, and COVL metrics for audio files.")
    parser.add_argument('--clean_folder', type=str, required=True, help='Path to the clean data folder')
    parser.add_argument('--noisy_folder', type=str, required=True, help='Path to the noisy data folder')
    parser.add_argument('--enhanced_folder', type=str, required=True, help='Path to the enhanced data folder')
    parser.add_argument('--output_csv', type=str, default='results.csv', help='Path to the output CSV file')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')

    args = parser.parse_args()

    # Set up logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    clean_folder = args.clean_folder
    enhanced_folder = args.enhanced_folder

    # Collect all filenames from the clean folder
    clean_files = sorted([f for f in os.listdir(clean_folder) if os.path.isfile(os.path.join(clean_folder, f))])
    logging.info(f"Found {len(clean_files)} files in clean folder.")

    results = []
    metrics_total = np.zeros(6)  # Since compute_metrics returns six metrics

    total_time = 0
    start_time = time.time()

    for filename in clean_files:
        clean_path = os.path.join(clean_folder, filename)
        enhanced_path = os.path.join(enhanced_folder, filename)

        # Check if corresponding files exist
        if not os.path.exists(enhanced_path):
            logging.warning(f"No matching enhanced file for {filename}, skipping.")
            continue

        try:
            # Load audio files
            s, sr_s = torchaudio.load(clean_path)
            s_hat, sr_s_hat = torchaudio.load(enhanced_path)

            # Ensure sample rates are the same
            if sr_s != sr_s_hat:
                logging.warning(f"Sample rates do not match for {filename}, resampling.")
                # Resample to 16000 Hz or 8000 Hz
                target_sr = 16000 if sr_s > 8000 else 8000
                resample_transform_s = torchaudio.transforms.Resample(orig_freq=sr_s, new_freq=target_sr)
                resample_transform_s_hat = torchaudio.transforms.Resample(orig_freq=sr_s_hat, new_freq=target_sr)
                s = resample_transform_s(s)
                s_hat = resample_transform_s_hat(s_hat)
                sr_s = sr_s_hat = target_sr

            # Ensure sample rate is acceptable for compute_metrics
            if sr_s not in [8000, 16000]:
                logging.warning(f"Sample rate {sr_s} Hz not supported. Skipping {filename}.")
                continue

            # Ensure single-channel
            if s.shape[0] > 1:
                logging.info(f"Converting clean signal to mono for {filename}.")
                s = torch.mean(s, dim=0, keepdim=True)
            if s_hat.shape[0] > 1:
                logging.info(f"Converting enhanced signal to mono for {filename}.")
                s_hat = torch.mean(s_hat, dim=0, keepdim=True)

            # Convert tensors to 1D numpy arrays
            s = s.squeeze().numpy()
            s_hat = s_hat.squeeze().numpy()

            # Ensure signals are the same length
            min_len = min(len(s), len(s_hat))
            s = s[:min_len]
            s_hat = s_hat[:min_len]

            # Normalize signals
            s = s / (np.max(np.abs(s)) + 1e-8)
            s_hat = s_hat / (np.max(np.abs(s_hat)) + 1e-8)

            # Call compute_metrics
            try:
                metrics = compute_metrics(s, s_hat, sr_s, path=0)
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                continue
            # metrics is a tuple: (pesq_mos, CSIG, CBAK, COVL, segSNR, STOI)
            # Convert to numpy array for accumulation
            metrics = np.array(metrics)
            metrics_total += metrics

            # Log results
            logging.info(f"{filename}: PESQ={metrics[0]:.3f}, CSIG={metrics[1]:.3f}, CBAK={metrics[2]:.3f}, COVL={metrics[3]:.3f}, SSNR={metrics[4]:.3f}, STOI={metrics[5]:.3f}")

            # Append to results
            results.append({
                'filename': filename,
                'PESQ': metrics[0],
                'CSIG': metrics[1],
                'CBAK': metrics[2],
                'COVL': metrics[3],
                'SSNR': metrics[4],
                'STOI': metrics[5],
            })

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue

    total_time = time.time() - start_time
    logging.info(f"Total time taken: {total_time:.2f} seconds")

    # Compute average metrics
    num_files = len(results)
    if num_files > 0:
        metrics_avg = metrics_total / num_files
        logging.info(f"Average PESQ: {metrics_avg[0]:.3f}")
        logging.info(f"Average CSIG: {metrics_avg[1]:.3f}")
        logging.info(f"Average CBAK: {metrics_avg[2]:.3f}")
        logging.info(f"Average COVL: {metrics_avg[3]:.3f}")
        logging.info(f"Average SSNR: {metrics_avg[4]:.3f}")
        logging.info(f"Average STOI: {metrics_avg[5]:.3f}")
    else:
        logging.info("No results to display.")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Results saved to {args.output_csv}")

if __name__ == '__main__':
    main()
