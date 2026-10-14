import functions_roman_rubin as frr


class DummyTelescope:
    def __init__(self, name):
        self.name = name


def _names(telescopes):
    return [
        tel.name
        for tel in telescopes
    ]


def test_generic_observation_mode_behavior_is_unchanged():
    # Historical generic helper remains untouched.
    assert (
        frr.observation_mode_passed(
            True,
            False,
            use_roman=True,
            use_rubin=True,
        )
        is False
    )


def test_scientific_selection_requires_only_roman_when_roman_is_active():
    # Roman available, Rubin unavailable -> selection may proceed.
    assert (
        frr.selection_data_available(
            True,
            False,
            use_roman=True,
            use_rubin=True,
        )
        is True
    )

    # Rubin cannot rescue absent Roman data.
    assert (
        frr.selection_data_available(
            False,
            True,
            use_roman=True,
            use_rubin=True,
        )
        is False
    )


def test_selection_telescopes_are_roman_only_when_roman_is_active():
    telescopes = [
        DummyTelescope("W149"),
        DummyTelescope("z"),
        DummyTelescope("i"),
    ]

    selected = frr.selection_telescopes(
        telescopes,
        use_roman=True,
        use_rubin=True,
    )

    assert _names(selected) == ["W149"]


def test_selection_telescopes_preserve_rubin_only_mode():
    telescopes = [
        DummyTelescope("W149"),
        DummyTelescope("z"),
        DummyTelescope("i"),
    ]

    selected = frr.selection_telescopes(
        telescopes,
        use_roman=False,
        use_rubin=True,
    )

    assert _names(selected) == ["z", "i"]


def test_sim_event_is_wired_to_selection_helpers():
    names = set(
        frr.sim_event.__code__.co_names
    )

    assert "selection_data_available" in names
    assert "selection_telescopes" in names
