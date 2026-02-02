from __future__ import annotations

import pandas as pd

from dr_wandb.utils import MAX_INT, convert_large_ints_in_data, safe_convert_for_parquet


class TestConvertLargeIntsInData:
    def test_returns_small_int_unchanged(self):
        assert convert_large_ints_in_data(100) == 100
        assert convert_large_ints_in_data(-100) == -100

    def test_converts_large_positive_int_to_float(self):
        large_int = MAX_INT + 1
        result = convert_large_ints_in_data(large_int)
        assert isinstance(result, float)
        assert result == float(large_int)

    def test_converts_large_negative_int_to_float(self):
        large_negative = -(MAX_INT + 1)
        result = convert_large_ints_in_data(large_negative)
        assert isinstance(result, float)
        assert result == float(large_negative)

    def test_handles_nested_dict(self):
        data = {"a": 1, "b": {"c": MAX_INT + 1, "d": 2}}
        result = convert_large_ints_in_data(data)
        assert result["a"] == 1
        assert isinstance(result["b"]["c"], float)
        assert result["b"]["d"] == 2

    def test_handles_nested_list(self):
        data = [1, MAX_INT + 1, [MAX_INT + 2, 3]]
        result = convert_large_ints_in_data(data)
        assert result[0] == 1
        assert isinstance(result[1], float)
        assert isinstance(result[2][0], float)
        assert result[2][1] == 3

    def test_handles_mixed_nested_structure(self):
        data = {"items": [{"value": MAX_INT + 1}, {"value": 5}]}
        result = convert_large_ints_in_data(data)
        assert isinstance(result["items"][0]["value"], float)
        assert result["items"][1]["value"] == 5

    def test_preserves_non_int_types(self):
        data = {"str": "hello", "float": 3.14, "none": None, "bool": True}
        result = convert_large_ints_in_data(data)
        assert result == data

    def test_boundary_value_at_max_int(self):
        assert convert_large_ints_in_data(MAX_INT) == MAX_INT
        assert isinstance(convert_large_ints_in_data(MAX_INT + 1), float)


class TestSafeConvertForParquet:
    def test_converts_large_int64_to_float64(self):
        df = pd.DataFrame({"col": [1, 2, MAX_INT + 1]})
        result = safe_convert_for_parquet(df)
        assert result["col"].dtype == "float64"

    def test_preserves_small_int64(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = safe_convert_for_parquet(df)
        assert result["col"].dtype == "int64"

    def test_converts_dict_in_object_column_to_json(self):
        df = pd.DataFrame({"col": [{"a": 1}, {"b": 2}]})
        result = safe_convert_for_parquet(df)
        assert result["col"].iloc[0] == '{"a": 1}'
        assert result["col"].iloc[1] == '{"b": 2}'

    def test_converts_list_in_object_column_to_json(self):
        df = pd.DataFrame({"col": [[1, 2], [3, 4]]})
        result = safe_convert_for_parquet(df)
        assert result["col"].iloc[0] == "[1, 2]"
        assert result["col"].iloc[1] == "[3, 4]"

    def test_converts_string_in_object_column(self):
        df = pd.DataFrame({"col": ["hello", "world"]})
        result = safe_convert_for_parquet(df)
        assert result["col"].iloc[0] == "hello"
        assert result["col"].iloc[1] == "world"

    def test_preserves_none_in_object_column(self):
        df = pd.DataFrame({"col": [{"a": 1}, None]})
        result = safe_convert_for_parquet(df)
        assert result["col"].iloc[0] == '{"a": 1}'
        assert result["col"].iloc[1] is None

    def test_handles_large_int_inside_dict(self):
        df = pd.DataFrame({"col": [{"big": MAX_INT + 1}]})
        result = safe_convert_for_parquet(df)
        assert f"{float(MAX_INT + 1)}" in result["col"].iloc[0]

    def test_does_not_modify_original_dataframe(self):
        df = pd.DataFrame({"col": [MAX_INT + 1]})
        original_dtype = df["col"].dtype
        safe_convert_for_parquet(df)
        assert df["col"].dtype == original_dtype
