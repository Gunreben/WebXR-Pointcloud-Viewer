#!/usr/bin/env python3
"""
Rewrite a .pcd or .ply point cloud with a new origin and orientation.

Examples
--------
python pointcloud_transform.py input.pcd output.pcd --origin 12.3 -1.4 0.0 --rotate 0 90 0
python pointcloud_transform.py input.ply output.ply --origin-mode bottom-center --rotate -90 0 180

Rotation uses the same XYZ Euler convention as the Three.js viewer:
rotate around X first, then Y, then Z.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, List, Sequence


def format_float(value: float) -> str:
    return format(value, ".9g")


def apply_xyz_rotation(point: Sequence[float], rotate_deg: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = point
    rx, ry, rz = [math.radians(v) for v in rotate_deg]

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # X rotation
    y, z = y * cx - z * sx, y * sx + z * cx
    # Y rotation
    x, z = x * cy + z * sy, -x * sy + z * cy
    # Z rotation
    x, y = x * cz - y * sz, x * sz + y * cz

    return x, y, z


def transform_point(
    point: Sequence[float],
    origin: Sequence[float],
    rotate_deg: Sequence[float],
    translate: Sequence[float],
) -> tuple[float, float, float]:
    shifted = (
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    )
    rotated = apply_xyz_rotation(shifted, rotate_deg)
    return (
        rotated[0] + translate[0],
        rotated[1] + translate[1],
        rotated[2] + translate[2],
    )


def update_bounds(bounds: list[list[float]], point: Sequence[float]) -> None:
    for axis, value in enumerate(point):
        if value < bounds[0][axis]:
            bounds[0][axis] = value
        if value > bounds[1][axis]:
            bounds[1][axis] = value


def midpoint(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def compute_origin_from_mode(
    mode: str,
    min_xyz: Sequence[float],
    max_xyz: Sequence[float],
) -> tuple[float, float, float]:
    center = midpoint(min_xyz, max_xyz)
    if mode == "center":
        return center
    if mode == "bottom-center":
        return (center[0], min_xyz[1], center[2])
    if mode == "min":
        return (min_xyz[0], min_xyz[1], min_xyz[2])
    raise ValueError(f"Unsupported origin mode: {mode}")


@dataclass
class PcdMetadata:
    header_bytes: bytes
    data_kind: str
    data_offset: int
    fields: list[str]
    sizes: list[int]
    types: list[str]
    counts: list[int]
    points: int
    width: int | None
    height: int | None


@dataclass
class PlyProperty:
    name: str
    is_list: bool
    value_type: str | None = None
    count_type: str | None = None
    item_type: str | None = None


@dataclass
class PlyElement:
    name: str
    count: int
    properties: list[PlyProperty]


@dataclass
class PlyMetadata:
    header_bytes: bytes
    data_offset: int
    fmt: str
    elements: list[PlyElement]


PCD_TYPE_MAP = {
    ("F", 4): "f",
    ("F", 8): "d",
    ("I", 1): "b",
    ("I", 2): "h",
    ("I", 4): "i",
    ("I", 8): "q",
    ("U", 1): "B",
    ("U", 2): "H",
    ("U", 4): "I",
    ("U", 8): "Q",
}

PLY_TYPE_MAP = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def read_all_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


def parse_pcd_metadata(data: bytes) -> PcdMetadata:
    marker = b"\nDATA "
    marker_index = data.find(marker)
    if marker_index < 0:
        if data.startswith(b"DATA "):
            marker_index = 0
        else:
            raise ValueError("PCD header is missing a DATA line.")

    end_of_data_line = data.find(b"\n", marker_index + 1)
    if end_of_data_line < 0:
        end_of_data_line = len(data)
        header_bytes = data
    else:
        end_of_data_line += 1
        header_bytes = data[:end_of_data_line]

    header_text = header_bytes.decode("ascii", errors="strict")
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    points = None
    width = None
    height = None
    data_kind = None

    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        key = key.upper()
        if key == "FIELDS":
            fields = value.split()
        elif key == "SIZE":
            sizes = [int(v) for v in value.split()]
        elif key == "TYPE":
            types = value.split()
        elif key == "COUNT":
            counts = [int(v) for v in value.split()]
        elif key == "POINTS":
            points = int(value)
        elif key == "WIDTH":
            width = int(value)
        elif key == "HEIGHT":
            height = int(value)
        elif key == "DATA":
            data_kind = value.strip().lower()

    if not fields:
        raise ValueError("PCD header is missing FIELDS.")
    if not sizes:
        raise ValueError("PCD header is missing SIZE.")
    if not types:
        raise ValueError("PCD header is missing TYPE.")
    if not counts:
        counts = [1] * len(fields)
    if len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths do not match.")
    if data_kind is None:
        raise ValueError("PCD header is missing DATA.")
    if data_kind == "binary_compressed":
        raise ValueError("PCD DATA binary_compressed is not supported by this helper.")
    if points is None:
        if width is not None and height is not None:
            points = width * height
        else:
            raise ValueError("PCD header is missing POINTS.")

    return PcdMetadata(
        header_bytes=header_bytes,
        data_kind=data_kind,
        data_offset=len(header_bytes),
        fields=fields,
        sizes=sizes,
        types=types,
        counts=counts,
        points=points,
        width=width,
        height=height,
    )


def pcd_column_layout(meta: PcdMetadata) -> tuple[list[str], list[str]]:
    names: list[str] = []
    fmt_parts: list[str] = []
    for field, size, kind, count in zip(meta.fields, meta.sizes, meta.types, meta.counts):
        code = PCD_TYPE_MAP.get((kind, size))
        if code is None:
            raise ValueError(f"Unsupported PCD field type: TYPE={kind} SIZE={size}")
        for idx in range(count):
            names.append(field if count == 1 else f"{field}[{idx}]")
            fmt_parts.append(code)
    return names, fmt_parts


def transform_pcd(
    input_path: Path,
    output_path: Path,
    origin_mode: str | None,
    origin_value: Sequence[float] | None,
    rotate_deg: Sequence[float],
    translate: Sequence[float],
) -> dict[str, object]:
    raw = read_all_bytes(input_path)
    meta = parse_pcd_metadata(raw)
    column_names, fmt_parts = pcd_column_layout(meta)
    xyz_indices = []
    for axis in ("x", "y", "z"):
        try:
            xyz_indices.append(column_names.index(axis))
        except ValueError as exc:
            raise ValueError("PCD file must contain x, y and z fields.") from exc

    bounds = [
        [float("inf"), float("inf"), float("inf")],
        [float("-inf"), float("-inf"), float("-inf")],
    ]
    data_bytes = raw[meta.data_offset:]

    if meta.data_kind == "ascii":
        body_text = data_bytes.decode("utf-8", errors="strict")
        rows = [line for line in body_text.splitlines() if line.strip()]
        if len(rows) != meta.points:
            raise ValueError(f"PCD points mismatch: header says {meta.points}, body has {len(rows)} rows.")

        parsed_rows: list[list[str]] = []
        for line in rows:
            parts = line.split()
            if len(parts) != len(column_names):
                raise ValueError(
                    f"PCD row has {len(parts)} values but expected {len(column_names)} from the header."
                )
            point = [float(parts[xyz_indices[0]]), float(parts[xyz_indices[1]]), float(parts[xyz_indices[2]])]
            update_bounds(bounds, point)
            parsed_rows.append(parts)

        origin = tuple(origin_value) if origin_value is not None else compute_origin_from_mode(origin_mode, bounds[0], bounds[1])

        out_lines: list[str] = []
        for parts in parsed_rows:
            point = [float(parts[xyz_indices[0]]), float(parts[xyz_indices[1]]), float(parts[xyz_indices[2]])]
            tx, ty, tz = transform_point(point, origin, rotate_deg, translate)
            parts[xyz_indices[0]] = format_float(tx)
            parts[xyz_indices[1]] = format_float(ty)
            parts[xyz_indices[2]] = format_float(tz)
            out_lines.append(" ".join(parts))

        output_path.write_bytes(meta.header_bytes + ("\n".join(out_lines) + "\n").encode("utf-8"))
    elif meta.data_kind == "binary":
        little_endian_fmt = "<" + "".join(fmt_parts)
        record_struct = struct.Struct(little_endian_fmt)
        stride = record_struct.size
        expected_size = stride * meta.points
        if len(data_bytes) < expected_size:
            raise ValueError(f"PCD binary payload is truncated: expected {expected_size} bytes, got {len(data_bytes)}.")

        rows: list[list[object]] = []
        for index in range(meta.points):
            start = index * stride
            values = list(record_struct.unpack_from(data_bytes, start))
            point = [float(values[xyz_indices[0]]), float(values[xyz_indices[1]]), float(values[xyz_indices[2]])]
            update_bounds(bounds, point)
            rows.append(values)

        origin = tuple(origin_value) if origin_value is not None else compute_origin_from_mode(origin_mode, bounds[0], bounds[1])

        output = bytearray(meta.header_bytes)
        for values in rows:
            point = [float(values[xyz_indices[0]]), float(values[xyz_indices[1]]), float(values[xyz_indices[2]])]
            tx, ty, tz = transform_point(point, origin, rotate_deg, translate)
            values[xyz_indices[0]] = tx
            values[xyz_indices[1]] = ty
            values[xyz_indices[2]] = tz
            output.extend(record_struct.pack(*values))
        output_path.write_bytes(output)
    else:
        raise ValueError(f"Unsupported PCD DATA type: {meta.data_kind}")

    return {
        "format": f"pcd/{meta.data_kind}",
        "points": meta.points,
        "origin": origin,
        "rotate_deg": tuple(rotate_deg),
        "translate": tuple(translate),
        "input": str(input_path),
        "output": str(output_path),
    }


def parse_ply_metadata(data: bytes) -> PlyMetadata:
    marker = b"end_header"
    idx = data.find(marker)
    if idx < 0:
        raise ValueError("PLY header is missing end_header.")

    line_end = data.find(b"\n", idx)
    if line_end < 0:
        line_end = len(data)
        header_bytes = data
    else:
        line_end += 1
        header_bytes = data[:line_end]

    header_text = header_bytes.decode("ascii", errors="strict")
    lines = header_text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError("File does not look like a PLY file.")

    fmt = None
    elements: list[PlyElement] = []
    current_element: PlyElement | None = None

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line or line.startswith("comment"):
            continue
        parts = line.split()
        if parts[0] == "format":
            if len(parts) < 2:
                raise ValueError("Invalid PLY format line.")
            fmt = parts[1]
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ValueError(f"Invalid PLY element line: {line}")
            current_element = PlyElement(parts[1], int(parts[2]), [])
            elements.append(current_element)
        elif parts[0] == "property":
            if current_element is None:
                raise ValueError("PLY property declared before any element.")
            if parts[1] == "list":
                if len(parts) != 5:
                    raise ValueError(f"Invalid PLY list property line: {line}")
                current_element.properties.append(
                    PlyProperty(parts[4], True, count_type=parts[2], item_type=parts[3])
                )
            else:
                if len(parts) != 3:
                    raise ValueError(f"Invalid PLY scalar property line: {line}")
                current_element.properties.append(PlyProperty(parts[2], False, value_type=parts[1]))

    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY format: {fmt}")

    return PlyMetadata(
        header_bytes=header_bytes,
        data_offset=len(header_bytes),
        fmt=fmt,
        elements=elements,
    )


def ply_struct_prefix(fmt: str) -> str:
    if fmt == "binary_little_endian":
        return "<"
    if fmt == "binary_big_endian":
        return ">"
    raise ValueError(f"PLY binary format expected, got: {fmt}")


def ply_scalar_format(type_name: str) -> str:
    code = PLY_TYPE_MAP.get(type_name)
    if code is None:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return code


def parse_ply_ascii_records(lines: list[str], meta: PlyMetadata) -> tuple[list[list[list[object]]], int]:
    records_by_element: list[list[list[object]]] = []
    line_index = 0

    for element in meta.elements:
        element_records: list[list[object]] = []
        for _ in range(element.count):
            if line_index >= len(lines):
                raise ValueError(f"PLY ended early while reading element '{element.name}'.")
            parts = lines[line_index].split()
            line_index += 1

            record: list[object] = []
            token_index = 0
            for prop in element.properties:
                if prop.is_list:
                    count = int(parts[token_index])
                    token_index += 1
                    items: list[object] = []
                    item_fmt = ply_scalar_format(prop.item_type)
                    is_float = item_fmt in {"f", "d"}
                    for _ in range(count):
                        token = parts[token_index]
                        token_index += 1
                        items.append(float(token) if is_float else int(token))
                    record.append(items)
                else:
                    fmt_code = ply_scalar_format(prop.value_type)
                    token = parts[token_index]
                    token_index += 1
                    record.append(float(token) if fmt_code in {"f", "d"} else int(token))

            if token_index != len(parts):
                raise ValueError(f"PLY line for element '{element.name}' has unexpected extra values.")
            element_records.append(record)
        records_by_element.append(element_records)

    return records_by_element, line_index


def serialize_ply_ascii(meta: PlyMetadata, records_by_element: list[list[list[object]]]) -> bytes:
    out_lines: list[str] = []
    for element, element_records in zip(meta.elements, records_by_element):
        for record in element_records:
            tokens: list[str] = []
            for prop, value in zip(element.properties, record):
                if prop.is_list:
                    items = list(value)
                    tokens.append(str(len(items)))
                    for item in items:
                        tokens.append(format_float(item) if isinstance(item, float) else str(item))
                else:
                    tokens.append(format_float(value) if isinstance(value, float) else str(value))
            out_lines.append(" ".join(tokens))
    body = "\n".join(out_lines)
    if body:
        body += "\n"
    return meta.header_bytes + body.encode("utf-8")


def parse_ply_binary_records(data: bytes, meta: PlyMetadata) -> list[list[list[object]]]:
    prefix = ply_struct_prefix(meta.fmt)
    offset = 0
    records_by_element: list[list[list[object]]] = []

    for element in meta.elements:
        element_records: list[list[object]] = []
        for _ in range(element.count):
            record: list[object] = []
            for prop in element.properties:
                if prop.is_list:
                    count_fmt = prefix + ply_scalar_format(prop.count_type)
                    count_size = struct.calcsize(count_fmt)
                    if offset + count_size > len(data):
                        raise ValueError(f"PLY ended early while reading list count in '{element.name}'.")
                    count = struct.unpack_from(count_fmt, data, offset)[0]
                    offset += count_size

                    item_fmt = prefix + ply_scalar_format(prop.item_type)
                    item_size = struct.calcsize(item_fmt)
                    items: list[object] = []
                    for _ in range(count):
                        if offset + item_size > len(data):
                            raise ValueError(f"PLY ended early while reading list items in '{element.name}'.")
                        items.append(struct.unpack_from(item_fmt, data, offset)[0])
                        offset += item_size
                    record.append(items)
                else:
                    value_fmt = prefix + ply_scalar_format(prop.value_type)
                    value_size = struct.calcsize(value_fmt)
                    if offset + value_size > len(data):
                        raise ValueError(f"PLY ended early while reading property '{prop.name}'.")
                    value = struct.unpack_from(value_fmt, data, offset)[0]
                    offset += value_size
                    record.append(value)
            element_records.append(record)
        records_by_element.append(element_records)

    if offset != len(data):
        trailing = len(data) - offset
        if trailing > 0:
            raise ValueError(f"PLY has {trailing} unexpected trailing bytes after parsing.")
    return records_by_element


def serialize_ply_binary(meta: PlyMetadata, records_by_element: list[list[list[object]]]) -> bytes:
    prefix = ply_struct_prefix(meta.fmt)
    out = bytearray(meta.header_bytes)

    for element, element_records in zip(meta.elements, records_by_element):
        for record in element_records:
            for prop, value in zip(element.properties, record):
                if prop.is_list:
                    count_pack = struct.pack(prefix + ply_scalar_format(prop.count_type), len(value))
                    out.extend(count_pack)
                    item_fmt = prefix + ply_scalar_format(prop.item_type)
                    for item in value:
                        out.extend(struct.pack(item_fmt, item))
                else:
                    value_fmt = prefix + ply_scalar_format(prop.value_type)
                    out.extend(struct.pack(value_fmt, value))
    return bytes(out)


def transform_vertex_records(
    meta: PlyMetadata,
    records_by_element: list[list[list[object]]],
    origin_mode: str | None,
    origin_value: Sequence[float] | None,
    rotate_deg: Sequence[float],
    translate: Sequence[float],
) -> tuple[tuple[float, float, float], int]:
    vertex_element_index = None
    for index, element in enumerate(meta.elements):
        if element.name == "vertex":
            vertex_element_index = index
            break
    if vertex_element_index is None:
        raise ValueError("PLY file has no 'vertex' element.")

    element = meta.elements[vertex_element_index]
    name_to_index = {prop.name: idx for idx, prop in enumerate(element.properties) if not prop.is_list}
    try:
        x_index = name_to_index["x"]
        y_index = name_to_index["y"]
        z_index = name_to_index["z"]
    except KeyError as exc:
        raise ValueError("PLY vertex element must contain scalar x, y and z properties.") from exc

    bounds = [
        [float("inf"), float("inf"), float("inf")],
        [float("-inf"), float("-inf"), float("-inf")],
    ]
    vertices = records_by_element[vertex_element_index]
    for row in vertices:
        point = [float(row[x_index]), float(row[y_index]), float(row[z_index])]
        update_bounds(bounds, point)

    origin = tuple(origin_value) if origin_value is not None else compute_origin_from_mode(origin_mode, bounds[0], bounds[1])

    for row in vertices:
        point = [float(row[x_index]), float(row[y_index]), float(row[z_index])]
        tx, ty, tz = transform_point(point, origin, rotate_deg, translate)
        row[x_index] = tx
        row[y_index] = ty
        row[z_index] = tz

    return origin, len(vertices)


def transform_ply(
    input_path: Path,
    output_path: Path,
    origin_mode: str | None,
    origin_value: Sequence[float] | None,
    rotate_deg: Sequence[float],
    translate: Sequence[float],
) -> dict[str, object]:
    raw = read_all_bytes(input_path)
    meta = parse_ply_metadata(raw)
    payload = raw[meta.data_offset:]

    if meta.fmt == "ascii":
        body_text = payload.decode("utf-8", errors="strict")
        lines = [line for line in body_text.splitlines() if line.strip()]
        records_by_element, consumed = parse_ply_ascii_records(lines, meta)
        if consumed != len(lines):
            raise ValueError("PLY ASCII parser did not consume the full file.")
        origin, points = transform_vertex_records(records_by_element=records_by_element, meta=meta, origin_mode=origin_mode, origin_value=origin_value, rotate_deg=rotate_deg, translate=translate)
        output_path.write_bytes(serialize_ply_ascii(meta, records_by_element))
    else:
        records_by_element = parse_ply_binary_records(payload, meta)
        origin, points = transform_vertex_records(records_by_element=records_by_element, meta=meta, origin_mode=origin_mode, origin_value=origin_value, rotate_deg=rotate_deg, translate=translate)
        output_path.write_bytes(serialize_ply_binary(meta, records_by_element))

    return {
        "format": f"ply/{meta.fmt}",
        "points": points,
        "origin": origin,
        "rotate_deg": tuple(rotate_deg),
        "translate": tuple(translate),
        "input": str(input_path),
        "output": str(output_path),
    }


def parse_triplet(values: Sequence[str], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} needs exactly 3 numbers.")
    return (float(values[0]), float(values[1]), float(values[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Change the origin and orientation of a .pcd or .ply point cloud."
    )
    parser.add_argument("input", type=Path, help="Input .pcd or .ply file")
    parser.add_argument("output", type=Path, help="Output .pcd or .ply file")
    origin_group = parser.add_mutually_exclusive_group()
    origin_group.add_argument(
        "--origin",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Source-space point that should become the new origin",
    )
    origin_group.add_argument(
        "--origin-mode",
        choices=("center", "bottom-center", "min"),
        default="bottom-center",
        help="Auto-pick the origin from the cloud bounds (default: bottom-center)",
    )
    parser.add_argument(
        "--rotate",
        nargs=3,
        default=("0", "0", "0"),
        metavar=("RX", "RY", "RZ"),
        help="Euler rotation in degrees, applied in XYZ order",
    )
    parser.add_argument(
        "--translate",
        nargs=3,
        default=("0", "0", "0"),
        metavar=("TX", "TY", "TZ"),
        help="Optional post-rotation translation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")

    if output_path.suffix.lower() != input_path.suffix.lower():
        parser.error("Input and output file extensions must match.")

    origin_value = parse_triplet(args.origin, "--origin") if args.origin else None
    rotate_deg = parse_triplet(args.rotate, "--rotate")
    translate = parse_triplet(args.translate, "--translate")

    try:
        suffix = input_path.suffix.lower()
        if suffix == ".pcd":
            summary = transform_pcd(
                input_path=input_path,
                output_path=output_path,
                origin_mode=None if origin_value is not None else args.origin_mode,
                origin_value=origin_value,
                rotate_deg=rotate_deg,
                translate=translate,
            )
        elif suffix == ".ply":
            summary = transform_ply(
                input_path=input_path,
                output_path=output_path,
                origin_mode=None if origin_value is not None else args.origin_mode,
                origin_value=origin_value,
                rotate_deg=rotate_deg,
                translate=translate,
            )
        else:
            parser.error("Only .pcd and .ply are supported.")
    except Exception as exc:  # pragma: no cover - CLI reporting
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Input:      {summary['input']}")
    print(f"Output:     {summary['output']}")
    print(f"Format:     {summary['format']}")
    print(f"Points:     {summary['points']}")
    print(
        "Origin:     "
        f"{format_float(summary['origin'][0])}, "
        f"{format_float(summary['origin'][1])}, "
        f"{format_float(summary['origin'][2])}"
    )
    print(
        "Rotate XYZ: "
        f"{format_float(summary['rotate_deg'][0])}, "
        f"{format_float(summary['rotate_deg'][1])}, "
        f"{format_float(summary['rotate_deg'][2])} deg"
    )
    print(
        "Translate:  "
        f"{format_float(summary['translate'][0])}, "
        f"{format_float(summary['translate'][1])}, "
        f"{format_float(summary['translate'][2])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
