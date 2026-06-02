"""Command-line entry point for cell2location.

Subcommands:
  list-skills        — show the bundled Claude / coding-agent skills.
  install-skills     — copy (or --symlink) the bundled skills into ~/.claude/skills/.
                       After running once, /spatial-mapping, /cell2location-context, and
                       /cell2location-troubleshooting appear in the slash-command menu of
                       any agent that respects ~/.claude/skills/ (Claude Code, Cursor,
                       Aider, Continue, Codex).
  uninstall-skills   — remove the previously installed copies/symlinks.

The bundled skills live next to the package at cell2location/_bundled_skills/, copied
into the wheel at build time from the repository's top-level .claude/skills/ directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_BUNDLED_DIR = _PACKAGE_DIR / "_bundled_skills"
# Editable-install fallback: when the package is installed via `pip install -e .` and
# the build-time mirror in setup.py has not run, the source .claude/skills/ tree is one
# level up from the package directory.
_DEV_FALLBACK_DIR = _PACKAGE_DIR.parent / ".claude" / "skills"
_INSTALL_PREFIX = "cell2location-"  # collision-avoidance + provenance


def _user_skills_dir() -> Path:
    """Resolved at call time so tests can patch $HOME without reloading the module."""
    return Path.home() / ".claude" / "skills"


def _resolve_skills_source() -> Path | None:
    if _BUNDLED_DIR.is_dir():
        return _BUNDLED_DIR
    if _DEV_FALLBACK_DIR.is_dir():
        return _DEV_FALLBACK_DIR
    return None


def _discover_bundled_skills() -> list[Path]:
    """Return one Path per bundled skill (each containing SKILL.md)."""
    source = _resolve_skills_source()
    if source is None:
        return []
    return sorted(p for p in source.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def _installed_name(skill_dir: Path) -> str:
    """Return the destination directory name under ~/.claude/skills/."""
    name = skill_dir.name
    if name.startswith("cell2location-"):
        return name  # already prefixed, e.g. cell2location-context, cell2location-troubleshooting
    return _INSTALL_PREFIX + name  # e.g. spatial-mapping -> cell2location-spatial-mapping


def _cmd_list_skills(_args: argparse.Namespace) -> int:
    skills = _discover_bundled_skills()
    source = _resolve_skills_source()
    if not skills:
        print(
            f"No bundled skills found. Checked:\n  - {_BUNDLED_DIR}\n  - {_DEV_FALLBACK_DIR}\n"
            "If you are running from an editable install (`pip install -e .`), the\n"
            "fallback path should exist; if it does not, you may be installing from a\n"
            "wheel that was built without the skills tree.",
            file=sys.stderr,
        )
        return 1
    user_skills_dir = _user_skills_dir()
    print(f"Bundled skills (source: {source}):")
    for skill in skills:
        dest = user_skills_dir / _installed_name(skill)
        status = "installed" if dest.exists() else "not installed"
        print(f"  - {skill.name:30s} -> {dest}  [{status}]")
    return 0


def _cmd_install_skills(args: argparse.Namespace) -> int:
    skills = _discover_bundled_skills()
    if not skills:
        print(
            f"No bundled skills found (looked in {_BUNDLED_DIR} and {_DEV_FALLBACK_DIR}).",
            file=sys.stderr,
        )
        return 1

    user_skills_dir = _user_skills_dir()
    if not args.dry_run:
        user_skills_dir.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for skill in skills:
        dest = user_skills_dir / _installed_name(skill)
        if dest.exists() or dest.is_symlink():
            if not args.force:
                print(f"Skipping {dest} (already exists; use --force to overwrite).")
                continue
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)

        if args.dry_run:
            mode = "symlink" if args.symlink else "copy"
            print(f"[dry-run] would {mode} {skill} -> {dest}")
            installed.append(dest.name)
            continue

        if args.symlink:
            dest.symlink_to(skill, target_is_directory=True)
        else:
            shutil.copytree(skill, dest)
        installed.append(dest.name)

    if installed:
        mode = "Linked" if args.symlink else "Installed"
        print(
            f"{mode} {len(installed)} skill(s) into {user_skills_dir}.\n"
            "Restart your coding agent (Claude Code, Cursor, Aider, …) to see the new\n"
            "slash commands: /spatial-mapping, /cell2location-context, "
            "/cell2location-troubleshooting."
        )
    else:
        print("Nothing installed. Use --force to overwrite existing entries.")
    return 0


def _cmd_uninstall_skills(args: argparse.Namespace) -> int:
    skills = _discover_bundled_skills()
    if not skills:
        print(
            f"No bundled skills found (looked in {_BUNDLED_DIR} and {_DEV_FALLBACK_DIR}); " "nothing to remove.",
            file=sys.stderr,
        )
        return 1

    user_skills_dir = _user_skills_dir()
    removed: list[str] = []
    for skill in skills:
        dest = user_skills_dir / _installed_name(skill)
        if not (dest.exists() or dest.is_symlink()):
            continue
        if args.dry_run:
            print(f"[dry-run] would remove {dest}")
            removed.append(dest.name)
            continue
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
        removed.append(dest.name)

    print(f"Removed {len(removed)} skill(s) from {user_skills_dir}.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cell2location",
        description="cell2location command-line utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list-skills", help="List the bundled coding-agent skills.")
    p_list.set_defaults(func=_cmd_list_skills)

    p_install = sub.add_parser(
        "install-skills",
        help="Install the bundled skills into ~/.claude/skills/.",
    )
    p_install.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink instead of copy (updates flow through `pip install -U cell2location`).",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing entries in ~/.claude/skills/.",
    )
    p_install.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be installed, without writing.",
    )
    p_install.set_defaults(func=_cmd_install_skills)

    p_uninstall = sub.add_parser(
        "uninstall-skills",
        help="Remove the previously installed cell2location-* skill entries.",
    )
    p_uninstall.add_argument("--dry-run", action="store_true", help="Print what would be removed.")
    p_uninstall.set_defaults(func=_cmd_uninstall_skills)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
