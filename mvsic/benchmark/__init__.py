"""Installed original-YAML build, search, and plotting workflows."""

from dataclasses import dataclass
from pathlib import Path

from .config import read_config, resolve_config


@dataclass(frozen=True)
class BenchmarkPlan:
    source: Path
    raw: dict
    base_dir: Path

    def validate(self, action=None, **options):
        from .legacy_plot import is_plot_config, validate_plot_config
        from .workflow import is_workflow, prepare
        options = {"base_dir": self.base_dir, **options}
        if is_plot_config(self.raw):
            if action not in (None, "plot"):
                raise ValueError("plot YAML requires benchmark plot or validate")
            return validate_plot_config(self.raw, base_dir=options["base_dir"])
        if is_workflow(self.raw):
            if action is None:
                indices = self.raw.get("indices", [])
                search = any(isinstance(index, dict) and ("builds" in index or any(
                    isinstance(build, dict) and build.get("search_configs")
                    for build in index.get("build_configs", []))) for index in indices)
                action = "search" if search else "build"
            jobs = prepare(self.raw, self.source, action, **options)
            return {"workflow": "original_yaml", "action": action, "jobs": len(jobs),
                    "operating_points": sum(len(job.get("points", [])) for job in jobs), "resolved": jobs}
        config = resolve_config(self.raw, self.source)
        return {"fingerprint": config.fingerprint, "operating_points": len(config.points()), "resolved": config.resolved}

    def run(self, action, **options):
        return run(action, self, **options)


def load_config(filename, *, base_dir=None):
    """Read an original YAML plan; relative paths use CWD or explicit base_dir."""
    source = Path(filename).expanduser().resolve(strict=True)
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    return BenchmarkPlan(source, read_config(source), root)


def run(action, config, **options):
    """Run build/search/plot from a loaded plan or a filename, without conversion."""
    from .legacy_plot import is_plot_config, plot_config
    from .workflow import is_workflow, run as run_workflow
    plan = config if isinstance(config, BenchmarkPlan) else load_config(config, base_dir=options.get("base_dir"))
    options = {"base_dir": plan.base_dir, **options}
    if action == "validate":
        return plan.validate(**options)
    if action not in ("build", "search", "plot"):
        raise ValueError("action must be validate, build, search or plot")
    if is_plot_config(plan.raw):
        plan.validate(action)
        return plot_config(plan.raw, **{key: options[key] for key in ("base_dir", "affix", "style", "file_format") if key in options})
    if is_workflow(plan.raw):
        return run_workflow(action, plan.raw, plan.source, **options)
    from .runner import run as run_job
    return run_job(action, resolve_config(plan.raw, plan.source), force=bool(options.get("force") or options.get("rebuild")))


__all__ = ["BenchmarkPlan", "load_config", "run"]
