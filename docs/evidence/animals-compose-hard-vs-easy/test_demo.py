from pathlib import Path

from PIL import Image

import demo


def test_all_six_source_triptychs_exist_on_disk():
    for quadrant, pair_slug, seed, _label in demo.ROWS:
        p = demo.cmp_path(quadrant, pair_slug, seed)
        assert p.exists(), f"expected real rendered file, missing: {p}"


def test_rate_labels_differ_between_hard_and_easy_pairs():
    # If this check can't fail, it isn't checking anything: assert the labels
    # for a known-hard pair and a known-easy pair are NOT the same string.
    compose_rate = demo.load_compose_rate()
    hard = demo.rate_label(compose_rate, "out_out", "a_frog__x__a_toad")
    easy = demo.rate_label(compose_rate, "out_out", "a_leopard__x__a_jaguar")
    assert hard != easy, "hard and easy pairs must not read identically"
    assert "0.9375" in hard
    assert "1.0000" in easy


def test_train_pairs_labeled_as_not_held_out():
    compose_rate = demo.load_compose_rate()
    label = demo.rate_label(compose_rate, "in_in", "a_wolf__x__a_husky")
    assert label == "training pair, not held out"


def test_missing_file_raises_not_silently_skips():
    bad_rows = list(demo.ROWS) + [("out_out", "a_nonexistent_pair", 99, "fake")]
    try:
        demo.build_figure(bad_rows)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing triptych")


def test_composed_figure_is_taller_than_a_single_tile():
    # Proves the 6 rows were actually stacked, not just one tile re-saved.
    single_tile = Image.open(demo.cmp_path(*demo.ROWS[0][:3]))
    fig = demo.build_figure(demo.ROWS)
    assert fig.size[1] > single_tile.size[1] * len(demo.ROWS) * 0.9
    assert fig.size[0] >= single_tile.size[0]


def test_composed_figure_width_accommodates_row_labels():
    single_tile = Image.open(demo.cmp_path(*demo.ROWS[0][:3]))
    fig = demo.build_figure(demo.ROWS)
    # Row-label column must add real width beyond the bare triptych.
    assert fig.size[0] > single_tile.size[0]


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
