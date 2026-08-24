"""Every named run family resolves, and every module still imports.

These are the checks the retrofit rename is judged against. The two existing test
files pin the sampler's numbers; neither of them would notice a constant pointing at a
directory that no longer exists, or an import left dangling by a package rename. That
gap is what let 103 code sites spell an ambiguous path for months without anything
going red.

No GPU, no model download. Both tests are filesystem and import only, so they run in
seconds and can gate every step of a rename rather than being saved for the end.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

from poe_repair import paths

REPO = Path(__file__).resolve().parent.parent

# Every public constant in paths.py that names a run family.
FAMILIES = sorted(
    name
    for name in dir(paths)
    if name.isupper()
    and not name.startswith("_")
    and name not in {"REPO_ROOT", "MOUNT_ROOT", "NOT_YET_PRODUCED"}
    and isinstance(getattr(paths, name), str)
)


def test_there_are_families_to_check():
    """A rename that empties the constant list would make every test below vacuous."""
    assert len(FAMILIES) >= 20, f"only {len(FAMILIES)} families found, expected the full set"


@pytest.mark.parametrize("family", FAMILIES)
def test_every_named_family_resolves(family):
    """A constant names somewhere real, on one filesystem or the other.

    Failure here means the name survived a move that the value did not, which is
    exactly the drift `paths.py` exists to stop. A family listed in `NOT_YET_PRODUCED`
    is exempt, because it declares where its code would write rather than where
    anything is.
    """
    rel = getattr(paths, family)
    holders = paths.roots_holding(rel)
    if family in paths.NOT_YET_PRODUCED:
        assert not holders, (
            f"{family} resolves now, so it has been produced. "
            f"Take it out of NOT_YET_PRODUCED."
        )
        return
    assert holders, (
        f"{family} = {rel!r} exists under neither {paths.REPO_ROOT} nor {paths.MOUNT_ROOT}"
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_resolve_returns_a_real_path(family):
    """`resolve` hands back something that exists, whichever root it picked."""
    if family in paths.NOT_YET_PRODUCED:
        pytest.skip(f"{family} declares a layout no run has produced")
    assert paths.resolve(getattr(paths, family)).exists()


def test_resolve_raises_rather_than_returning_a_path_that_is_not_there():
    with pytest.raises(FileNotFoundError):
        paths.resolve("outputs/a-family-that-was-never-run")


def test_resolve_rejects_an_unknown_preference():
    with pytest.raises(ValueError):
        paths.resolve(paths.WINDOW, prefer="whichever")


def _package_modules():
    """Every importable module under poe_repair, including subpackages."""
    import poe_repair

    for info in pkgutil.walk_packages(poe_repair.__path__, prefix="poe_repair."):
        if "__pycache__" in info.name:
            continue
        yield info.name


@pytest.mark.parametrize("module", sorted(_package_modules()))
def test_every_module_imports(module):
    """No dangling import survives a package rename.

    This is the check that would have caught a half-finished rename of the twelve
    packages under `poe_repair/experiments/`, where 194 import sites name them.
    """
    try:
        importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(f"{module} does not import: {exc}")
    except Exception as exc:  # noqa: BLE001 - a module that raises on import is also broken
        pytest.fail(f"{module} raised {type(exc).__name__} on import: {exc}")


def test_no_module_spells_a_bare_output_path():
    """After the rename, paths come from `paths.py` and are not spelled by hand.

    Marked xfail until the rename lands, so it records the target without failing the
    suite meanwhile. Flip it to a plain assertion when the sweep is done: at that point
    a new hand-written path is a regression, and this is what says so.
    """
    offenders = []
    for py in (REPO / "poe_repair").rglob("*.py"):
        if "__pycache__" in py.parts or py.name == "paths.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "/datasets/mmolefe/poe_repair_min" in text:
            offenders.append(py.relative_to(REPO))
    if offenders:
        pytest.xfail(
            f"{len(offenders)} modules still hardcode the mount path; "
            f"the rename has not landed yet"
        )
