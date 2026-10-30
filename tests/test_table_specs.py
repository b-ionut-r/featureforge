"""Table specs: Windows drive letters stay in the path, and DataFrames can stand in for paths."""
import pandas as pd

from featureforge.pipeline import _table_specs, split_spec


def test_windows_drive_letter_is_part_of_the_path():
    assert split_spec(r"att=D:\NFL\att.parquet:nfl_id") == ("att", [r"D:\NFL\att.parquet", "nfl_id"])
    assert split_spec("att=C:/data/att.csv:nfl_id:t") == ("att", ["C:/data/att.csv", "nfl_id", "t"])
    assert split_spec(r"att=D:\NFL\att.parquet") == ("att", [r"D:\NFL\att.parquet"])
    assert split_spec("att=data/att.parquet:nfl_id") == ("att", ["data/att.parquet", "nfl_id"])


def test_dataframe_tables_are_written_to_parquet(tmp_path):
    df = pd.DataFrame({"nfl_id": [1, 1, 2], "v": [0.1, 0.2, 0.3]})
    (spec,) = _table_specs({"att": (df, "nfl_id")}, str(tmp_path))
    name, parts = split_spec(spec)
    assert name == "att" and parts[1] == "nfl_id"
    pd.testing.assert_frame_equal(pd.read_parquet(parts[0]), df)
