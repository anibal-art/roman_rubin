from analysis.fisher_information import (
    _resolve_physical_derivative_method,
)


class DummyModel:

    def __init__(
        self,
        model_type,
    ):
        self._model_type = model_type

    def model_type(self):
        return self._model_type


def test_auto_uses_chain_rule_for_fsplarge():

    model = DummyModel(
        "FSPLarge"
    )

    assert (
        _resolve_physical_derivative_method(
            model,
            "auto",
        )
        == "fspl_chain_rule"
    )


def test_auto_does_not_use_vbm_chain_rule_for_yoo_fspl():

    model = DummyModel(
        "FSPL"
    )

    assert (
        _resolve_physical_derivative_method(
            model,
            "auto",
        )
        == "five_point_stable_pair"
    )


def test_auto_uses_five_point_for_pspl():

    model = DummyModel(
        "PSPL"
    )

    assert (
        _resolve_physical_derivative_method(
            model,
            "auto",
        )
        == "five_point_stable_pair"
    )


def test_explicit_method_is_not_overridden():

    model = DummyModel(
        "PSPL"
    )

    assert (
        _resolve_physical_derivative_method(
            model,
            "fspl_chain_rule",
        )
        == "fspl_chain_rule"
    )

    assert (
        _resolve_physical_derivative_method(
            model,
            "five_point_stable_pair",
        )
        == "five_point_stable_pair"
    )
