"""Event-realization layer: decide WHAT to simulate, before simulating it.

`functions_roman_rubin.sim_event` is the legacy adapter: it keeps its
historical signature (required for internal and LRT compatibility) but
delegates the decision of blending/caustic-origin values to this package
before calling pyLIMA's model/flux-parameter constructors itself.
"""
