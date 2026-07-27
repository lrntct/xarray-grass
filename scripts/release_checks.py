from __future__ import annotations

import argparse
import ast
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

PROJECT_NAME = "xarray-grass"
VERSION_FILE = Path("src/xarray_grass/__init__.py")


def version_from_tag(tag: str) -> str:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise SystemExit(f"Release tags must use the vX.Y.Z format, got {tag!r}.")
    return tag[1:]


def version_from_source(version_file: Path) -> str:
    module = ast.parse(version_file.read_text(), filename=str(version_file))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            if isinstance(value, str):
                return value
    raise SystemExit(f"Could not find a string __version__ in {version_file}.")


def verify_version(tag: str, version_file: Path) -> None:
    tag_version = version_from_tag(tag)
    source_version = version_from_source(version_file)
    if source_version != tag_version:
        raise SystemExit(
            f"Tag {tag!r} does not match package version {source_version!r}."
        )


def verify_metadata(metadata_bytes: bytes, artifact: Path, version: str) -> None:
    metadata = BytesParser().parsebytes(metadata_bytes)
    if metadata["Name"] != PROJECT_NAME or metadata["Version"] != version:
        raise SystemExit(
            f"{artifact} has {metadata['Name']} {metadata['Version']}, "
            f"expected {PROJECT_NAME} {version}."
        )


def verify_artifacts(dist: Path, tag: str) -> None:
    version = version_from_tag(tag)
    sdists = list(dist.glob("*.tar.gz"))
    wheels = list(dist.glob("*.whl"))

    if len(sdists) != 1:
        raise SystemExit(f"Expected one sdist, found {len(sdists)}: {sdists}")
    if len(wheels) != 1:
        raise SystemExit(f"Expected one wheel, found {len(wheels)}: {wheels}")

    with tarfile.open(sdists[0]) as archive:
        metadata_files = [
            member
            for member in archive.getmembers()
            if PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(metadata_files) != 1:
            raise SystemExit(f"Could not identify sdist metadata in {sdists[0]}.")
        metadata_file = archive.extractfile(metadata_files[0])
        if metadata_file is None:
            raise SystemExit(f"Could not read sdist metadata in {sdists[0]}.")
        verify_metadata(metadata_file.read(), sdists[0], version)

    with zipfile.ZipFile(wheels[0]) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise SystemExit(f"Could not identify wheel metadata in {wheels[0]}.")
        verify_metadata(archive.read(metadata_files[0]), wheels[0], version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate release versions and artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("verify-version")
    version_parser.add_argument("tag")
    version_parser.add_argument("--version-file", type=Path, default=VERSION_FILE)

    artifacts_parser = subparsers.add_parser("verify-artifacts")
    artifacts_parser.add_argument("dist", type=Path)
    artifacts_parser.add_argument("tag")

    args = parser.parse_args()
    if args.command == "verify-version":
        verify_version(args.tag, args.version_file)
    else:
        verify_artifacts(args.dist, args.tag)


if __name__ == "__main__":
    main()
