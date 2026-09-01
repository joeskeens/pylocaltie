#!/usr/bin/python
"""
==============================================================================
  This file is part of the VDIF2RINEX software package.

  This is free software; you can redistribute it and/or modify
  it under the terms of the BSD 3-Clause License. See the LICENSE file
  distributed with this software package for the full license text.

  We are distributing this in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  BSD 3-Clause License for more details.

  This software was developed by Applied Research Laboratories at the
  University of Texas at Austin.

  Copyright 2026, The Board of Regents of The University of Texas System
==============================================================================
"""
import numpy as np
import argparse
import struct
from numba import njit
from vdif_tools import VDIFHeader, VDIFStats, get_vdif_stats, unpack_vdif_chunk

def add_args_to_parser(parser):
    """ Add arguments to parser """
    parser.add_argument("-i_x", dest="input_files_x", action="append", type=str, nargs="+", help = 'Input VDIF X polarization file(s) to track.')
    parser.add_argument("-i_y", dest="input_files_y", action="append", type=str, nargs="+", help = 'Input VDIF Y polarization file(s) to track.')
    parser.add_argument("-o", dest="output_files", action="append", type=str, nargs="+", help = 'Output RINEX file(s) (same length as VDIF).')

@njit(cache=True)
def get_sample_words(i_out, q_out, samples_per_chunk, bit_depth):
    '''Packs interleaved I/Q samples into 32-bit VDIF data words'''
    samples_per_word = 32 // (bit_depth * 2)
    n_words = samples_per_chunk // samples_per_word
    words = np.empty(n_words, dtype=np.uint32)
    for w in range(n_words):
        word = np.uint32(0)
        for s in range(samples_per_word):
            idx = w * samples_per_word + s
            nibble = np.uint32((q_out[idx] << bit_depth) | i_out[idx])
            word |= nibble << np.uint32(s * bit_depth * 2)
        words[w] = word
    return words

@njit(cache=True)
def unpack_vdif_chunk(buf, start, buf_len, target_thread,
                      bits_per_sample, bps_component, bit_mask,
                      is_complex, samples_per_chunk, header_size, word_size):
    """
    Read VDIF frames from buf starting at byte offset `start`, unpack
    samples from frames matching target_thread, until samples_per_chunk
    samples are collected or the buffer is exhausted.

    vdif_combine gets its own unpack_vdif_chunk because we do not iq_threshold 
    and we combine all channels rather than extracting just one

    Returns (i_out, q_out, bytes_consumed).
    """
    i_out = np.empty(samples_per_chunk, dtype=np.uint32)
    q_out = np.empty(samples_per_chunk, dtype=np.uint32)
    s = 0
    pos = start

    while s < samples_per_chunk and pos + header_size <= buf_len:
        # -- parse thread_id from header word 2, bits [25:16] --
        # word 2 (bytes 8-11): data_frame_len
        w2 = (np.uint32(buf[pos + 8])
              | (np.uint32(buf[pos + 9]) << 8)
              | (np.uint32(buf[pos + 10]) << 16)
              | (np.uint32(buf[pos + 11]) << 24))
        frame_len = np.int64(w2 & np.uint32(0x00FFFFFF)) * 8
        
        # word 3 (bytes 12-15): thread_id
        w3 = (np.uint32(buf[pos + 12])
              | (np.uint32(buf[pos + 13]) << 8)
              | (np.uint32(buf[pos + 14]) << 16)
              | (np.uint32(buf[pos + 15]) << 24))
        thread_id = (w3 >> 16) & np.uint32(0x3FF)

        if frame_len <= header_size or pos + frame_len > buf_len:
            break

        # -- skip wrong thread --
        if thread_id != np.uint32(target_thread):
            pos += frame_len
            continue

        # -- unpack data payload word by word --
        data_start = pos + header_size
        data_end = pos + frame_len
        wp = data_start
          
        while wp + word_size <= data_end and s < samples_per_chunk:
            word = (np.uint32(buf[wp])
                    | (np.uint32(buf[wp + 1]) << 8)
                    | (np.uint32(buf[wp + 2]) << 16)
                    | (np.uint32(buf[wp + 3]) << 24))
            wp += word_size

            bits_left = np.int32(32)
            while bits_left >= bits_per_sample and s < samples_per_chunk:
                i_out[s] = word & np.uint32(bit_mask)
                word >>= np.uint32(bps_component)

                if is_complex:
                    q_out[s] = word & np.uint32(bit_mask)
                    word >>= np.uint32(bps_component)

                bits_left -= bits_per_sample
                s += 1
        pos += frame_len

    return i_out[:s], q_out[:s], pos - start

def process_vdif(vdif_files, vdif_files_dual, output_files):
    """ Read VDIF files, track GNSS signals """
    for idx, vdif_file_handle in enumerate(vdif_files):
        output_file = output_files[idx]
        first_frame = True

        # we assume one source per VDIF file 
        vdif_file = np.memmap(vdif_file_handle, dtype=np.uint8, mode='r')
        print(f'processing file {vdif_file_handle.split("/")[-1]}')

        # get VDIF file setup
        thread=0
        vdif_stats, header = get_vdif_stats(vdif_file, thread, None)
        sample_rate = (vdif_stats.frames_per_sec * vdif_stats.data_bytes_per_frame * 8) \
            // vdif_stats.bits_per_sample
        consumed = 0
        total_bytes = vdif_file.shape[0]

        IS_COMPLEX = vdif_stats.is_complex
        BITS_PER_SAMPLE = vdif_stats.bits_per_sample
        BIT_MASK = (1 << vdif_stats.bits_per_sample_component) - 1
        frame_len = header.data_frame_len
        header_size = VDIFHeader.size()
        word_size = VDIFHeader.WORD_SIZE

        vdif_file_handle_dual = vdif_files_dual[idx]
        vdif_file_dual = np.memmap(vdif_file_handle_dual, dtype=np.uint8, mode='r')
        print(f'processing file {vdif_file_handle_dual.split("/")[-1]}')

        # get VDIF file setup
        thread_dual = thread+1
        vdif_stats_dual, header_dual = get_vdif_stats(vdif_file_dual, thread_dual, None)
        consumed_dual = 0
        total_bytes_dual = vdif_file_dual.shape[0]
        frame_len_dual = header.data_frame_len

        # Create VDIF header
        complex_frame_len = vdif_stats.data_bytes_per_frame*2 + header_size
        header_init = {
            'station_id' : str(header.station_id),
            'frame_no' : header.frame_no,
            'data_frame_len' : complex_frame_len,
            'data_type' : 'complex',
            'bits_per_sample' : BITS_PER_SAMPLE,
            'thread_id' : thread,
            'frames_per_sec' : vdif_stats.frames_per_sec,
            'ref_epoch': header.ref_epoch,
            'seconds_from_ref_epoch': header.seconds_from_ref_epoch,
            'frame_size': complex_frame_len,
            'num_channels': vdif_stats.num_channels
        }
        header_out = VDIFHeader(**header_init)

        start_sec = header.seconds_from_ref_epoch + header.frame_no/vdif_stats.frames_per_sec
        start_sec_dual = header_dual.seconds_from_ref_epoch + header_dual.frame_no/vdif_stats.frames_per_sec
        frame_diff = int(np.round((start_sec - start_sec_dual)*vdif_stats.frames_per_sec))
        if frame_diff - np.round((start_sec - start_sec_dual)*vdif_stats.frames_per_sec) != 0:
            breakpoint()
        if frame_diff > 0:
            # mismatch of start time -- need to align
            consumed_dual += frame_diff*frame_len_dual
        elif frame_diff < 0:
            consumed += abs(frame_diff)*frame_len

        #samples_per_chunk = vdif_stats.data_bytes_per_frame # int(sample_rate * acq_cadence)
        samples_per_frame = (vdif_stats.data_bytes_per_frame * 8) // BITS_PER_SAMPLE
        samples_per_chunk = samples_per_frame  # or samples_per_frame * N
        # process VDIF file  
        with open(output_file, 'wb') as fout:
            frames = 0 
            while consumed < total_bytes:
                count_samples = 0
                i_out = []
                q_out = []
                i_out, _, bytes_read = unpack_vdif_chunk(
                        vdif_file, consumed, total_bytes, thread,
                        BITS_PER_SAMPLE, vdif_stats.bits_per_sample_component, BIT_MASK,
                        IS_COMPLEX, samples_per_chunk, header_size, word_size
                    )
                consumed += bytes_read
                q_out, _, bytes_read = unpack_vdif_chunk(
                        vdif_file_dual, consumed_dual, total_bytes_dual, thread_dual,
                        BITS_PER_SAMPLE, vdif_stats_dual.bits_per_sample_component, BIT_MASK,
                        IS_COMPLEX, samples_per_chunk, header_size, word_size
                    )
                consumed_dual += bytes_read
                samples_combined = get_sample_words(i_out, q_out, samples_per_chunk, BITS_PER_SAMPLE)
                data_array = samples_combined.tobytes()
                array_len = len(data_array)

                # Update seconds field and create data frame
                hdr_words = b''.join(struct.pack(
                    '<I', header_out.word(i)) for i in range(0, VDIFHeader.NUM_WORDS))

                # Finally write header and data to file
                fout.write(hdr_words)
                fout.write(data_array)

                # Prepare next VDIF header
                header_out.increment_frame()
                frames+=1

                if consumed >= total_bytes: break
                if consumed_dual >= total_bytes_dual: break
        print(f"frames {frames}")
        print(f"Total bytes processed={consumed+consumed_dual}")

def main():
    """
    """
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()

    if args.input_files_x is not None and args.input_files_y is not None:
        process_vdif(args.input_files_x[0], args.input_files_y[0], args.output_files[0])
    else:
        raise ValueError('Need to supply X + Y polarization input files')


if __name__ == "__main__":
    main()

