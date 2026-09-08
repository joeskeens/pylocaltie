#!/usr/bin/env python3
"""
merge_rinex_obs.py -- merge RINEX 3 observation files that cover the same
epochs but carry different observables.

The intended case is one file per signal (E1, E6, E5a, ...) written out by a
front end that emits a separate RINEX per band, which need to be combined into
a single file whose SYS / # / OBS TYPES record is the union of the inputs.

Merging is done on the RINEX text rather than through gnsstk, because the
observation vector in Rinex3ObsData is indexed positionally against the header
obs-type list, so a merge is fundamentally a re-indexing of fixed-width fields.

Usage:
    merge_rinex_obs.py -o merged.rnx E1.rnx E6.rnx [E5a.rnx ...]

Observable ordering in the output follows input order: all of file 1's types
for a system, then any new types from file 2, and so on. Where two inputs carry
the same observable for the same satellite and epoch, the first non-blank value
wins and the collision is reported.
"""

import argparse
import sys
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

OBS_WIDTH = 16          # F14.3 + LLI + SSI
BLANK_OBS = " " * OBS_WIDTH
LABEL_COL = 60          # header label starts here (0-based)

# Header records regenerated from the merge rather than copied from file 1.
REGENERATED = {
    "SYS / # / OBS TYPES",
    "TIME OF FIRST OBS",
    "TIME OF LAST OBS",
    "PGM / RUN BY / DATE",
    "END OF HEADER",
    "# OF SATELLITES",
    "PRN / # OF OBS",
}

# Header records collected from every input and de-duplicated, because they
# name individual observation codes and so are not file-1-only properties.
MERGED_RECORDS = ["SYS / PHASE SHIFT", "SYS / SCALE FACTOR", "GLONASS COD/PHS/BIS"]


class RinexError(Exception):
    pass


def label_of(line):
    return line[LABEL_COL:].rstrip()


def body_of(line):
    return line[:LABEL_COL]


def hdr_line(body, label):
    return "{:<60}{:<20}".format(body[:60], label).rstrip()


def parse_epoch_time(line):
    """Epoch record: '>' then Y M D H M S. Returns a datetime."""
    try:
        year = int(line[2:6])
        month = int(line[6:9])
        day = int(line[9:12])
        hour = int(line[12:15])
        minute = int(line[15:18])
        sec = float(line[18:29])
    except ValueError as exc:
        raise RinexError("malformed epoch record: {!r}".format(line)) from exc
    whole = int(sec)
    micro = int(round((sec - whole) * 1e6))
    return datetime(year, month, day, hour, minute) + timedelta(
        seconds=whole, microseconds=micro
    )


def format_epoch(t, flag, nsat, clock_offset):
    line = "> {:4d} {:02d} {:02d} {:02d} {:02d}{:11.7f}  {:1d}{:3d}".format(
        t.year, t.month, t.day, t.hour, t.minute,
        t.second + t.microsecond / 1e6, flag, nsat,
    )
    if clock_offset:
        line += clock_offset
    return line.rstrip()


def read_rinex(path):
    """Return (header_lines, obs_types, epochs, time_system).

    obs_types maps system char -> list of 3-char codes.
    epochs maps datetime -> {'flag': int, 'clock': str, 'sats': {sat: [fields]}}.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()

    header, obs_types, time_system = [], OrderedDict(), None
    idx = 0
    for idx, line in enumerate(lines):
        label = label_of(line)
        header.append(line)
        if label == "END OF HEADER":
            break
    else:
        raise RinexError("{}: no END OF HEADER".format(path))
    data_start = idx + 1

    version_line = header[0] if header else ""
    if version_line[:9].strip() and not version_line[:9].strip().startswith("3"):
        print(
            "warning: {} declares version {!r}; this tool assumes RINEX 3".format(
                path, version_line[:9].strip()
            ),
            file=sys.stderr,
        )

    # SYS / # / OBS TYPES, with continuation lines (blank system column).
    current_sys = None
    for line in header:
        if label_of(line) != "SYS / # / OBS TYPES":
            continue
        body = body_of(line)
        sys_char = body[0].strip()
        if sys_char:
            current_sys = sys_char
            obs_types.setdefault(current_sys, [])
        if current_sys is None:
            raise RinexError("{}: continuation before any system".format(path))
        for start in range(7, 60, 4):
            code = body[start:start + 3].strip()
            if code:
                obs_types[current_sys].append(code)
        if sys_char:
            declared = body[3:6].strip()
            if declared:
                obs_types[current_sys + "_n"] = int(declared)

    counts = {k: v for k, v in obs_types.items() if k.endswith("_n")}
    for key in counts:
        obs_types.pop(key)
    for sys_char, expected in ((k[0], v) for k, v in counts.items()):
        got = len(obs_types.get(sys_char, []))
        if got != expected:
            raise RinexError(
                "{}: system {} declares {} obs types, found {}".format(
                    path, sys_char, expected, got
                )
            )

    for line in header:
        if label_of(line) == "TIME OF FIRST OBS":
            time_system = body_of(line)[48:51].strip() or None

    # Data records.
    epochs = OrderedDict()
    i = data_start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if not line.startswith(">"):
            raise RinexError(
                "{}: expected epoch record at line {}, got {!r}".format(path, i + 1, line)
            )
        padded = line.ljust(56)
        flag = int(padded[31:32] or "0")
        nsat = int(padded[32:35])
        clock = padded[35:56].rstrip()
        t = parse_epoch_time(padded)

        if flag > 1:
            print(
                "warning: {}: skipping epoch flag {} block at {}".format(path, flag, t),
                file=sys.stderr,
            )
            i += 1 + nsat
            continue

        sats = OrderedDict()
        for j in range(nsat):
            i += 1
            if i >= len(lines):
                raise RinexError("{}: truncated epoch at {}".format(path, t))
            rec = lines[i]
            sat = rec[:3].strip()
            if not sat:
                raise RinexError("{}: blank satellite id at line {}".format(path, i + 1))
            if len(sat) == 2:      # tolerate 'E9' for 'E09'
                sat = sat[0] + "0" + sat[1]
            n_types = len(obs_types.get(sat[0], []))
            padded_rec = rec[3:].ljust(n_types * OBS_WIDTH)
            fields = [
                padded_rec[k * OBS_WIDTH:(k + 1) * OBS_WIDTH]
                for k in range(n_types)
            ]
            sats[sat] = fields
        epochs[t] = {"flag": flag, "clock": clock, "sats": sats}
        i += 1

    return header, obs_types, epochs, time_system


def merge(paths, tol_seconds):
    parsed = [read_rinex(p) for p in paths]

    # Union of observables per system, in input order.
    merged_types = OrderedDict()
    for _, obs_types, _, _ in parsed:
        for sys_char, codes in obs_types.items():
            slot = merged_types.setdefault(sys_char, [])
            for code in codes:
                if code not in slot:
                    slot.append(code)

    # Bucket epochs onto a common grid so that files that disagree in the last
    # digit of the seconds field still line up.
    def key(t):
        if tol_seconds <= 0:
            return t
        quantum = timedelta(seconds=tol_seconds)
        return datetime.min + round((t - datetime.min) / quantum) * quantum

    merged = {}
    collisions = 0
    for path, (_, obs_types, epochs, _) in zip(paths, parsed):
        for t, rec in epochs.items():
            k = key(t)
            slot = merged.setdefault(
                k, {"time": t, "flag": rec["flag"], "clock": rec["clock"], "sats": {}}
            )
            if not slot["clock"]:
                slot["clock"] = rec["clock"]
            for sat, fields in rec["sats"].items():
                target = slot["sats"].setdefault(
                    sat, [BLANK_OBS] * len(merged_types[sat[0]])
                )
                for code, value in zip(obs_types[sat[0]], fields):
                    col = merged_types[sat[0]].index(code)
                    if value.strip():
                        if target[col].strip():
                            collisions += 1
                            if collisions <= 5:
                                print(
                                    "warning: {} {} {} already set; keeping first "
                                    "value ({})".format(
                                        sat, code, t, path
                                    ),
                                    file=sys.stderr,
                                )
                        else:
                            target[col] = value

    if collisions > 5:
        print(
            "warning: {} duplicate observations in total".format(collisions),
            file=sys.stderr,
        )
    return parsed, merged_types, merged


def build_header(parsed, merged_types, times, time_system):
    template = parsed[0][0]
    out = []

    # Observable records for each system, built up front so they can be placed
    # where the first input carried them.
    obs_records = []
    for sys_char, codes in merged_types.items():
        for chunk_start in range(0, len(codes), 13):
            chunk = codes[chunk_start:chunk_start + 13]
            if chunk_start == 0:
                body = "{:1s}  {:3d}".format(sys_char, len(codes))
            else:
                body = " " * 6
            body += "".join(" {:3s}".format(c) for c in chunk)
            obs_records.append(hdr_line(body, "SYS / # / OBS TYPES"))

    seen_merged = set()
    placed_obs = False
    for line in template:
        label = label_of(line)
        if label == "SYS / # / OBS TYPES":
            if not placed_obs:
                out.extend(obs_records)
                placed_obs = True
            continue
        if label in REGENERATED:
            continue
        if label in MERGED_RECORDS:
            continue
        if label == "RINEX VERSION / TYPE":
            out.append(line)
            out.append(
                hdr_line(
                    "{:<20}{:<40}".format(
                        "merge_rinex_obs.py",
                        datetime.now(timezone.utc).strftime("%Y%m%d %H%M%S UTC"),
                    ),
                    "PGM / RUN BY / DATE",
                )
            )
            continue
        out.append(line)

    if not placed_obs:
        out.extend(obs_records)

    # Records carried over from every input, de-duplicated on the body text.
    for label in MERGED_RECORDS:
        for header, _, _, _ in parsed:
            for line in header:
                if label_of(line) != label:
                    continue
                sig = (label, body_of(line).rstrip())
                if sig in seen_merged:
                    continue
                seen_merged.add(sig)
                out.append(line)

    ts = time_system or "GPS"
    for t, label in ((min(times), "TIME OF FIRST OBS"), (max(times), "TIME OF LAST OBS")):
        body = "{:6d}{:6d}{:6d}{:6d}{:6d}{:13.7f}{:5s}{:<3s}".format(
            t.year, t.month, t.day, t.hour, t.minute,
            t.second + t.microsecond / 1e6, "", ts,
        )
        out.append(hdr_line(body, label))

    out.append(hdr_line("", "END OF HEADER"))
    return out


def write_output(path, header, merged, merged_types):
    with open(path, "w") as fh:
        for line in header:
            fh.write(line.ljust(60 + len(label_of(line))).rstrip() + "\n")
        for k in sorted(merged):
            rec = merged[k]
            sats = sorted(rec["sats"])
            fh.write(
                format_epoch(rec["time"], rec["flag"], len(sats), rec["clock"]) + "\n"
            )
            for sat in sats:
                fields = rec["sats"][sat]
                fh.write((sat + "".join(fields)).rstrip() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("inputs", nargs="+", help="RINEX 3 observation files")
    ap.add_argument("-o", "--output", required=True, help="merged output file")
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="epoch matching tolerance in seconds (default 1e-4; 0 for exact)",
    )
    args = ap.parse_args(argv)

    if len(args.inputs) < 2:
        print("warning: only one input file given", file=sys.stderr)

    parsed, merged_types, merged = merge(args.inputs, args.tol)
    if not merged:
        print("error: no observations found", file=sys.stderr)
        return 1

    time_system = next((p[3] for p in parsed if p[3]), None)
    times = [rec["time"] for rec in merged.values()]
    header = build_header(parsed, merged_types, times, time_system)
    write_output(args.output, header, merged, merged_types)

    for sys_char, codes in merged_types.items():
        print(
            "{}: {} observables -> {}".format(sys_char, len(codes), " ".join(codes)),
            file=sys.stderr,
        )
    print(
        "{} epochs written to {}".format(len(merged), args.output), file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
