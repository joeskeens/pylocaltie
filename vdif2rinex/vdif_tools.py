#!/usr/bin/python
"""
==============================================================================
  This file is part of the VDIF2RINEX software package.

  This is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 3.0 of the License, or
  any later version.

  We are distributing this in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License; if not, write to the Free Software Foundation,
  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110, USA

  This software was developed by Applied Research Laboratories at the
  University of Texas at Austin.

  Copyright 2026, The Board of Regents of The University of Texas System
==============================================================================
"""
import pytz
import math
import struct
from collections import namedtuple
from numba import njit
from datetime import datetime, timedelta, timezone
import numpy as np

VDIFStats = namedtuple("VDIFStats",
                       ["total_frames",
                        "frames_per_sec",
                        "frame_len",
                        "data_bytes_per_frame",
                        "bits_per_sample",
                        "bits_per_sample_component",
                        "is_complex",
                        "samples_per_frame",
                        "file_start_utc",
                        "num_channels"
                        ])

STATION_ID_DEFAULT = 'Tt'
class VDIFHeader(object):
    '''Class for representing a VDIF Header'''
    NUM_WORDS = 8
    WORD_SIZE = 4 # bytes
    def __init__(self, **kwargs):
        # Word 0
        self.invalid_data = kwargs.get('invalid_data', 0)
        self.legacy_mode = kwargs.get('legacy_mode', 0)
        self.seconds_from_ref_epoch = kwargs.get('seconds_from_ref_epoch', 0)
        # Word 1
        self.ref_epoch = kwargs.get('ref_epoch', 0)
        self.frame_no = kwargs.get('frame_no', 0)
        # Word 2
        self.vdif_version = kwargs.get('vdif_version', 0)
        self.num_channels = kwargs.get('num_channels', 1)
        self.data_frame_len = kwargs.get('data_frame_len', 0)
        # Word 3
        self.data_type = kwargs.get('data_type', 'real')
        self.bits_per_sample = kwargs.get('bits_per_sample', 1)
        self.thread_id = kwargs.get('thread_id', 0)
        self.station_id = kwargs.get('station_id', STATION_ID_DEFAULT)

        self.frames_per_sec = kwargs.get('frames_per_sec', None)
        if not self.frames_per_sec:
            raise RuntimeError('Need frames per second')

    def unpack(self, buf):
        '''Unpack a buffer of bytes'''
        if len(buf) < self.size():
            raise ValueError('Invalid length of header')

        words = struct.unpack('i' * self.NUM_WORDS, buf[:self.size()])
        # Word 0
        self.invalid_data = (words[0] >> 31) & 0x1
        self.legacy_mode = (words[0] >> 30) & 0x1
        self.seconds_from_ref_epoch = words[0] & 0x3fffffff
        # Word 1
        self.ref_epoch = (words[1] >> 24) & 0x3f
        self.frame_no = words[1] & 0xffffff
        # Word 2
        self.vdif_version = (words[2] >> 29) & 0x7
        self.num_channels = 2 ** ((words[2] >> 24) & 0x1f)
        self.data_frame_len = 8 * (words[2] & 0xffffff)
        # Word 3
        self.data_type = 'complex' if ((words[3] >> 31) & 0x1) else 'real'
        self.bits_per_sample = ((words[3] >> 26) & 0x1f) + 1
        self.thread_id = (words[3] >> 16) & 0x3ff
        self.station_id = words[3] & 0xffff

    def __repr__(self):
        return (f"Word 0:\n"
                f" Invalid data:    {self.invalid_data!r}\n"
                f" Legacy mode:     {self.legacy_mode!r}\n"
                f"Word 1:\n"
                f" Reference epoch: {self.ref_epoch!r}\n"
                f" Data frame #:    {self.frame_no!r}\n"
                f"Word 2:\n"
                f" Version:         {self.vdif_version!r}\n"
                f" # Channels:      {self.num_channels!r}\n"
                f" Data frame len:  {self.data_frame_len!r}\n"
                f"Word 3:\n"
                f" Data type:       {self.data_type!r}\n"
                f" # bits/sample:   {self.bits_per_sample!r}\n"
                f" Thread ID:       {self.thread_id!r}\n"
                f" Station ID:      {self.station_id!r}\n")

    @classmethod
    def size(cls):
        '''Return header size in bytes'''
        return cls.NUM_WORDS * cls.WORD_SIZE

    def word(self, word_no):
        '''Generate VDIF Header Word'''
        word = 0
        if word_no == 0:
            word |= (self.invalid_data & 0x1) << 31
            word |= (self.legacy_mode & 0x1) << 30
            word |= self.seconds_from_ref_epoch & 0x3fffffff
        elif word_no == 1:
            word |= (self.ref_epoch & 0x3f) << 24
            word |= self.frame_no & 0xffffff
        elif word_no == 2:
            word |= (self.vdif_version & 0x7) << 29
            word |= (int(math.log(self.num_channels, 2)) & 0x1f) << 24
            word |= self.data_frame_len // 8
        elif word_no == 3:
            is_complex = 0 if self.data_type == 'real' else 1
            word |= is_complex << 31
            word |= ((self.bits_per_sample - 1) & 0x1f) << 26
            word |= (self.thread_id & 0x3ff) << 16
            # Station ID
            first_letter = self.station_id[0]
            second_letter = self.station_id[1]
            word |= ord(first_letter) << 8
            word |= ord(second_letter)
        elif word_no >= self.NUM_WORDS:
            raise ValueError('Invalid VDIF Word number')

        return word

    def increment_frame(self):
        '''Increments the frame number and wraps if necessary'''
        self.frame_no += 1
        if self.frame_no == self.frames_per_sec:
            self.frame_no = 0

def get_vdif_stats(buf, thread, num_channels):
    """Get basic stats from VDIF file.
    
    Parameters
    ----------
    buf : np.ndarray, dtype=uint8
        Memory-mapped VDIF file, e.g. np.memmap(path, dtype=np.uint8, mode='r')
    thread : int
        Target thread ID
    """
    total_bytes = buf.shape[0]
    consumed = 0
    header = VDIFHeader(**{"frames_per_sec": 1})
    header.unpack(bytes(buf[consumed:consumed + VDIFHeader.size()]))
    header_frame_last = header.frame_no - 1
    bits_per_sample_component = header.bits_per_sample
    bits_per_sample = header.bits_per_sample
    if num_channels is None:
        num_channels = header.num_channels
    is_complex = header.data_type != "real"
    if is_complex:
        bits_per_sample *= 2
    total_data_bytes = 0
    frames_per_sec = 0
    total_frames = 0
    data_bytes = header.data_frame_len - VDIFHeader.size()
    years, months = divmod(6 * header.ref_epoch, 12)
    dt = (datetime(2000 + years, months + 1, 1, tzinfo=pytz.utc) +
          timedelta(seconds=header.seconds_from_ref_epoch))
    while consumed < total_bytes:
        header.unpack(
            bytes(buf[consumed:consumed + VDIFHeader.size()]))
        frame_len = header.data_frame_len
        if header.thread_id == thread:
            dbytes = frame_len - VDIFHeader.size()
            if data_bytes != dbytes:
                raise ValueError("Inconsistent number of data bytes in frame")
            if header.invalid_data:
                raise ValueError("VDIF has an invalid frame!")
            if header.frame_no != header_frame_last + 1 and header.frame_no != 0:
                raise ValueError("Frame discontinuity!")
            data_bytes = dbytes
            total_data_bytes += data_bytes
            if header.frame_no > frames_per_sec:
                frames_per_sec = header.frame_no
            total_frames += 1
            header_frame_last = header.frame_no
            #if header.seconds_from_ref_epoch > 74850:
            #    print(header.frame_no)
            #    print(header.seconds_from_ref_epoch)
        consumed += frame_len
    frames_per_sec += 1
    stats = VDIFStats(
        total_frames=total_frames,
        frames_per_sec=frames_per_sec,
        frame_len=frame_len,
        data_bytes_per_frame=data_bytes,
        bits_per_sample=bits_per_sample,
        bits_per_sample_component=bits_per_sample_component,
        is_complex=is_complex,
        num_channels=num_channels,
        samples_per_frame=data_bytes * 8 // (bits_per_sample * num_channels),
        file_start_utc=dt)
    header.unpack(bytes(buf[:VDIFHeader.size()])) # hold beginning in header
    return stats, header

@njit(cache=True)
def iq_thres_map(sample, bits_per_sample):
    if bits_per_sample == 2:
        sign = np.int8((sample >> 1) * 2 - 1)   # -1 or +1
        mag  = np.int8(1 + 2 * (sample & 1))    #  1 or  3
        return np.int8(sign * mag)
    if bits_per_sample == 1 and sample == 0:
        return np.int8(-1)
    return np.int8(1)

@njit(cache=True)
def unpack_vdif_chunk(buf, start, buf_len, target_thread,
                      bits_per_sample, bps_component, bit_mask,
                      is_complex, samples_per_chunk, header_size, word_size,
                      num_channels, chan_mask):
    """
    Read VDIF frames from buf starting at byte offset `start`, unpack
    samples from frames matching target_thread and target_channel,
    until samples_per_chunk samples are collected or the buffer is exhausted.
    Returns (i_out, q_out, bytes_consumed).
    """
    i_out = np.empty(samples_per_chunk, dtype=np.int8)
    q_out = np.empty(samples_per_chunk, dtype=np.int8)
    s = 0
    pos = start
    while s < samples_per_chunk and pos + header_size <= buf_len:
        w2 = (np.uint32(buf[pos + 8])
              | (np.uint32(buf[pos + 9]) << 8)
              | (np.uint32(buf[pos + 10]) << 16)
              | (np.uint32(buf[pos + 11]) << 24))
        frame_len = np.int64(w2 & np.uint32(0x00FFFFFF)) * 8

        w3 = (np.uint32(buf[pos + 12])
              | (np.uint32(buf[pos + 13]) << 8)
              | (np.uint32(buf[pos + 14]) << 16)
              | (np.uint32(buf[pos + 15]) << 24))
        thread_id = (w3 >> 16) & np.uint32(0x3FF)
        if frame_len <= header_size or pos + frame_len > buf_len:
            break
        if thread_id != np.uint32(target_thread):
            pos += frame_len
            continue

        data_start = pos + header_size
        data_end = pos + frame_len
        wp = data_start
        chan = np.uint8(0)

        while wp + word_size <= data_end and s < samples_per_chunk:
            word = (np.uint32(buf[wp])
                    | (np.uint32(buf[wp + 1]) << 8)
                    | (np.uint32(buf[wp + 2]) << 16)
                    | (np.uint32(buf[wp + 3]) << 24))
            wp += word_size
            bits_left = np.int32(32)
            while bits_left >= bits_per_sample and s < samples_per_chunk:
                code = word & np.uint32(bit_mask)
                val_i = iq_thres_map(code, bps_component)
                word >>= np.uint32(bps_component)
                if is_complex:
                    code_q = word & np.uint32(bit_mask)
                    val_q = iq_thres_map(code_q, bps_component)
                    word >>= np.uint32(bps_component)
                else:
                    val_q = np.int8(0.0)
                bits_left -= bits_per_sample

                if chan_mask[chan]:
                    i_out[s] = val_i
                    q_out[s] = val_q
                    s += 1

                chan += np.uint8(1)
                if chan == num_channels:
                    chan = np.uint8(0)

        pos += frame_len
    i_out = i_out[:s]
    q_out = q_out[:s]
    return i_out, q_out, pos - start
