"""Tests for data.context assay type inference."""

import pandas as pd
import pytest

from LNPBO.data.context import ASSAY_TYPES, add_assay_type, infer_assay_type_row


class TestInferAssayTypeRow:
    """Parametrized test covering all 5 assay type classification paths."""

    @pytest.mark.parametrize(
        "row_data, expected",
        [
            # In vitro single formulation
            (
                {"Model": "in_vitro", "Route_of_administration": "", "Model_target": "", "Experiment_batching": ""},
                "in_vitro_single_formulation",
            ),
            # In vitro barcode screen
            (
                {"Model": "in_vitro", "Route_of_administration": "", "Model_target": "", "Experiment_batching": "barcoded"},
                "in_vitro_barcode_screen",
            ),
            # In vivo liver
            (
                {"Model": "in_vivo", "Route_of_administration": "intravenous", "Model_target": "liver", "Experiment_batching": ""},
                "in_vivo_liver",
            ),
            # In vivo other (intramuscular, lung)
            (
                {"Model": "in_vivo", "Route_of_administration": "intramuscular", "Model_target": "lung", "Experiment_batching": ""},
                "in_vivo_other",
            ),
            # Unknown (empty fields)
            (
                {"Model": "", "Route_of_administration": "", "Model_target": "", "Experiment_batching": ""},
                "unknown",
            ),
        ],
        ids=["in_vitro_single", "in_vitro_barcode", "in_vivo_liver", "in_vivo_other", "unknown"],
    )
    def test_assay_classification(self, row_data, expected):
        assert infer_assay_type_row(pd.Series(row_data)) == expected

    def test_route_infers_in_vivo_when_model_missing(self):
        row = pd.Series({"Model": "", "Route_of_administration": "intravenous", "Model_target": "spleen", "Experiment_batching": ""})
        assert infer_assay_type_row(row) == "in_vivo_other"

    def test_nan_handling(self):
        row = pd.Series({"Model": None, "Route_of_administration": None, "Model_target": None, "Experiment_batching": None})
        assert infer_assay_type_row(row) == "unknown"

    def test_nan_float_handling(self):
        row = pd.Series({"Model": float("nan"), "Route_of_administration": float("nan"), "Model_target": "", "Experiment_batching": ""})
        assert infer_assay_type_row(row) == "unknown"


class TestAddAssayType:
    def test_preserves_original_and_adds_column(self):
        df = pd.DataFrame({
            "Model": ["in_vitro", "in_vivo"],
            "Route_of_administration": ["", "intravenous"],
            "Model_target": ["", "liver"],
            "Experiment_batching": ["", ""],
        })
        result = add_assay_type(df)
        assert "assay_type" not in df.columns
        assert "assay_type" in result.columns
        assert result["assay_type"].iloc[0] == "in_vitro_single_formulation"
        assert result["assay_type"].iloc[1] == "in_vivo_liver"
