"""Observation dataset loading utilities.

The project consumes point-wise observations for the inverse PINN. This module
adds support for real datasets stored as CSV/TSV/text tables or NPZ archives,
with either positional columns or named columns. For text tables, a standard
CSV header row is detected automatically when present.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _normalize_format(path: Path, format_name: str | None) -> str:
    normalized = str(format_name or "auto").strip().lower()
    if normalized != "auto":
        return normalized

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "npz"
    if suffix in {".csv", ".tsv", ".txt"}:
        return "csv"
    return "csv"


def _as_float_vector(value: Sequence[float] | float | None, *, name: str, default: float) -> np.ndarray:
    if value is None:
        return np.full(3, float(default), dtype=np.float64)

    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.repeat(array, 3)
    if array.size != 3:
        raise ValueError(f"{name} must be a scalar or a length-3 sequence")
    return array


def _pick_column_index(fieldnames: list[str], column_name: str) -> int:
    try:
        return fieldnames.index(column_name)
    except ValueError as exc:
        available = ", ".join(fieldnames)
        raise KeyError(f"Column '{column_name}' was not found. Available columns: {available}") from exc


def _is_numeric_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _infer_text_table_header_mode(
    path: Path,
    *,
    delimiter: str,
    skiprows: int,
    comment_prefix: str | None,
) -> bool:
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle.readlines()[skiprows:]:
            stripped = line.strip()
            if not stripped:
                continue
            if comment_prefix and stripped.startswith(comment_prefix):
                continue

            reader = csv.reader([line], delimiter=delimiter)
            try:
                cells = [cell.strip() for cell in next(reader)]
            except StopIteration:
                continue

            meaningful_cells = [cell for cell in cells if cell]
            if not meaningful_cells:
                continue

            return not all(_is_numeric_token(cell) for cell in meaningful_cells)

    raise ValueError(f"Observation file is empty after filtering comments and skipped rows: {path}")


def _load_text_table_with_header(
    path: Path,
    *,
    delimiter: str,
    skiprows: int,
    comment_prefix: str | None,
    point_columns: Sequence[int],
    value_column: int,
    point_column_names: Sequence[str] | None,
    value_column_name: str | None,
    drop_invalid_rows: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_handle:
        lines = file_handle.readlines()[skiprows:]

    filtered_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if comment_prefix and stripped.startswith(comment_prefix):
            continue
        filtered_lines.append(line)

    if not filtered_lines:
        raise ValueError(f"Observation file is empty after filtering comments and skipped rows: {path}")

    reader = csv.reader(StringIO("".join(filtered_lines)), delimiter=delimiter)
    try:
        header = [cell.strip() for cell in next(reader)]
    except StopIteration as exc:
        raise ValueError(f"Observation file does not contain a header row: {path}") from exc

    if point_column_names is not None:
        point_names = [str(name) for name in point_column_names]
    else:
        point_names = [header[int(index)] for index in point_columns]

    if value_column_name is not None:
        target_name = str(value_column_name)
    else:
        target_name = header[int(value_column)]

    point_indices = [_pick_column_index(header, name) for name in point_names]
    value_index = _pick_column_index(header, target_name)

    points_rows: list[list[float]] = []
    values_rows: list[list[float]] = []
    for row_index, row in enumerate(reader, start=2):
        if not row:
            continue
        try:
            point_row = [float(row[column_index]) for column_index in point_indices]
            value_row = [float(row[value_index])]
        except (IndexError, TypeError, ValueError):
            if drop_invalid_rows:
                continue
            raise ValueError(f"Invalid numeric values in row {row_index} of {path}")

        points_rows.append(point_row)
        values_rows.append(value_row)

    points = np.asarray(points_rows, dtype=np.float64)
    values = np.asarray(values_rows, dtype=np.float64)
    if points.size == 0 or values.size == 0:
        raise ValueError(f"No valid observations were loaded from {path}")

    metadata = {
        "format": "csv",
        "path": str(path),
        "rows": int(points.shape[0]),
        "header": header,
        "point_names": point_names,
        "value_name": target_name,
        "point_indices": [int(index) for index in point_indices],
        "value_index": int(value_index),
    }
    return points, values, metadata


def _load_text_table_numeric(
    path: Path,
    *,
    delimiter: str,
    skiprows: int,
    comment_prefix: str | None,
    point_columns: Sequence[int],
    value_column: int,
    drop_invalid_rows: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    table = np.genfromtxt(
        path,
        delimiter=delimiter,
        comments=comment_prefix,
        skip_header=skiprows,
        dtype=np.float64,
        ndmin=2,
        invalid_raise=not drop_invalid_rows,
    )

    if table.size == 0:
        raise ValueError(f"Observation file is empty: {path}")

    if table.ndim != 2:
        table = np.atleast_2d(table)

    required_columns = max(max(point_columns), int(value_column)) + 1
    if table.shape[1] < required_columns:
        raise ValueError(
            f"Observation file must have at least {required_columns} columns; got {table.shape[1]} in {path}"
        )

    if drop_invalid_rows:
        finite_mask = np.all(np.isfinite(table[:, list(point_columns) + [int(value_column)]]), axis=1)
        table = table[finite_mask]

    if table.shape[0] == 0:
        raise ValueError(f"No valid observation rows were loaded from {path}")

    points = np.asarray(table[:, list(point_columns)], dtype=np.float64)
    values = np.asarray(table[:, int(value_column) : int(value_column) + 1], dtype=np.float64)
    metadata = {
        "format": "csv",
        "path": str(path),
        "rows": int(points.shape[0]),
        "point_indices": [int(index) for index in point_columns],
        "value_index": int(value_column),
    }
    return points, values, metadata


def _load_npz_archive(
    path: Path,
    *,
    npz_point_key: str,
    npz_value_key: str,
    coordinate_scale: Sequence[float] | float | None,
    coordinate_offset: Sequence[float] | float | None,
    value_scale: float,
    value_offset: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        if npz_point_key not in data.files:
            raise KeyError(f"NPZ archive {path} does not contain a '{npz_point_key}' array")

        value_key_candidates = [npz_value_key, "potential", "values", "value", "u", "resistivity", "apparent_resistivity"]
        point_array = np.asarray(data[npz_point_key], dtype=np.float64)
        value_key = next((key for key in value_key_candidates if key in data.files), None)
        if value_key is None:
            raise KeyError(
                f"NPZ archive {path} does not contain a usable value array. Available arrays: {data.files}"
            )
        value_array = np.asarray(data[value_key], dtype=np.float64)

    if point_array.ndim != 2 or point_array.shape[1] not in (3, 12):
        raise ValueError(f"Point array in {path} must have shape (N, 3) or (N, 12); got {point_array.shape}")

    points = point_array.copy()
    values = np.asarray(value_array, dtype=np.float64).reshape(-1, 1)
    if points.shape[0] != values.shape[0]:
        raise ValueError(
            f"Point and value arrays must have the same number of rows in {path}; got {points.shape[0]} and {values.shape[0]}"
        )

    scale = _as_float_vector(coordinate_scale, name="coordinate_scale", default=1.0)
    offset = _as_float_vector(coordinate_offset, name="coordinate_offset", default=0.0)
    if points.shape[1] == 12:
        scale = np.tile(scale, 4)
        offset = np.tile(offset, 4)
    points = points * scale + offset
    values = values * float(value_scale) + float(value_offset)

    metadata = {
        "format": "npz",
        "path": str(path),
        "rows": int(points.shape[0]),
        "point_key": npz_point_key,
        "value_key": str(value_key),
    }
    return points, values, metadata


def load_observation_arrays(
    path: str | Path,
    *,
    format: str = "auto",
    delimiter: str = ",",
    skiprows: int = 0,
    comment_prefix: str | None = "#",
    has_header: bool = False,
    point_columns: Sequence[int] = (0, 1, 2),
    value_column: int = 3,
    point_column_names: Sequence[str] | None = None,
    value_column_name: str | None = None,
    coordinate_scale: Sequence[float] | float | None = None,
    coordinate_offset: Sequence[float] | float | None = None,
    value_scale: float = 1.0,
    value_offset: float = 0.0,
    npz_point_key: str = "points",
    npz_value_key: str = "potential",
    drop_invalid_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load observation points and values from a real or synthetic dataset.

    The loader accepts either a CSV-like text table or an NPZ archive. For text
    files, observations can be selected by positional columns or by column names.
    CSV headers are inferred automatically when the first data row is textual.
    """

    file_path = Path(path)
    resolved_format = _normalize_format(file_path, format)

    if resolved_format == "npz":
        return _load_npz_archive(
            file_path,
            npz_point_key=npz_point_key,
            npz_value_key=npz_value_key,
            coordinate_scale=coordinate_scale,
            coordinate_offset=coordinate_offset,
            value_scale=value_scale,
            value_offset=value_offset,
        )

    use_header = bool(has_header or point_column_names is not None or value_column_name is not None)
    if not use_header:
        use_header = _infer_text_table_header_mode(
            file_path,
            delimiter=delimiter,
            skiprows=skiprows,
            comment_prefix=comment_prefix,
        )

    if use_header:
        points, values, metadata = _load_text_table_with_header(
            file_path,
            delimiter=delimiter,
            skiprows=skiprows,
            comment_prefix=comment_prefix,
            point_columns=point_columns,
            value_column=value_column,
            point_column_names=point_column_names,
            value_column_name=value_column_name,
            drop_invalid_rows=drop_invalid_rows,
        )
    else:
        points, values, metadata = _load_text_table_numeric(
            file_path,
            delimiter=delimiter,
            skiprows=skiprows,
            comment_prefix=comment_prefix,
            point_columns=point_columns,
            value_column=value_column,
            drop_invalid_rows=drop_invalid_rows,
        )

    scale = _as_float_vector(coordinate_scale, name="coordinate_scale", default=1.0)
    offset = _as_float_vector(coordinate_offset, name="coordinate_offset", default=0.0)
    if points.shape[1] == 12:
        scale = np.tile(scale, 4)
        offset = np.tile(offset, 4)
    points = points * scale + offset
    values = values * float(value_scale) + float(value_offset)

    metadata["format"] = resolved_format
    metadata["header_mode"] = use_header
    metadata["skiprows"] = int(skiprows)
    metadata["delimiter"] = delimiter
    return points, values, metadata
