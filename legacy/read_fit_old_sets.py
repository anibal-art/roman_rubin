"""LEGACY - not used by production. Kept for reference only.

read_fit re-fitted events saved in the old set_sim<n>/Event_<n>.h5 layout
without passing the event RA/Dec, so the fit used the fixed field
coordinates (removed from fit_lc.py: RA_FIELD = 267.92497054815516,
DEC_FIELD = -29.152232510353276). Fits now require the event coordinates.
Not imported anywhere.
"""


def read_fit(
    nsource,
    nset,
    path_run,
    model,
    algo,
    path_to_save_fit,
    path_ephemerides,
    fit_model=None,
    fit_parallax=True,
):

    path_event = path_run + f'/set_sim{nset}/Event_{nsource}.h5'

    print('path_event', path_event)
    print('os.path.getsize(path_model):', os.path.getsize(path_event))
    if os.path.getsize(path_event) == 0:
        return
    else:
        model, fit_model, fit_parallax = resolve_fit_options(
            model,
            fit_model=fit_model,
            fit_parallax=fit_parallax,
            default_fit_parallax=True,
        )

        info_event, pyLIMA_parameters, curves = read_data(path_event)

        lc_to_fit = {}
        for telo in curves:
            if len(curves[telo]['mag']) != 0:
                df = curves[telo][['time', 'mag', 'err_mag']].to_pandas()
                lc_to_fit[telo] = df.values
            else:
                lc_to_fit[telo] = []

        origin = info_event[2]
        rango = 1

        print("Start the fit using Roman and Rubin data:")
        fit_rr, event_fit_rr, pyLIMAmodel_rr = fit_rubin_roman(
            nsource, pyLIMA_parameters, path_to_save_fit, path_ephemerides,
            model, algo, origin, rango,
            lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
            lc_to_fit["i"], lc_to_fit["z"], lc_to_fit["y"],
            fit_model=fit_model,
            fit_parallax=fit_parallax,
        )
        print("Start the fit using only the Roman data:")
        fit_roman, event_fit_roman, pyLIMAmodel_roman = fit_rubin_roman(
            nsource, pyLIMA_parameters, path_to_save_fit, path_ephemerides,
            model, algo, origin, rango,
            lc_to_fit["W149"], [], [], [], [], [], [],
            fit_model=fit_model,
            fit_parallax=fit_parallax,
        )


# ---- test removed from tests/test_regression_interfaces.py ----
def test_read_fit_calls_fit_rubin_roman_unqualified():
    assert frr.read_fit.__globals__["fit_rubin_roman"] is frr.fit_rubin_roman

    sentinel = object()
    original = frr.fit_rubin_roman
    frr.fit_rubin_roman = sentinel
    try:
        assert frr.read_fit.__globals__["fit_rubin_roman"] is sentinel
    finally:
        frr.fit_rubin_roman = original
