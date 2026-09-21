from pydantic import BaseModel


class PackageVersionResult(BaseModel):
    package: str
    installed_version: str
    latest_version: str
    up_to_date: bool
    message: str


def get_package_version(
    package: str,
) -> PackageVersionResult:
    """
    Deterministic mock tool for checking
    installed and latest package versions.
    """

    mock_packages = {
        "transformers": {
            "installed_version": "4.56.0",
            "latest_version": "4.56.1",
        },
        "torch": {
            "installed_version": "2.8.0",
            "latest_version": "2.8.0",
        },
        "pydantic": {
            "installed_version": "2.11.7",
            "latest_version": "2.11.7",
        },
        "langgraph": {
            "installed_version": "0.6.6",
            "latest_version": "0.6.7",
        },
    }

    package_name = package.lower().strip()

    data = mock_packages.get(
        package_name,
        {
            "installed_version": "unknown",
            "latest_version": "unknown",
        },
    )

    installed_version = data["installed_version"]
    latest_version = data["latest_version"]

    up_to_date = (
        installed_version == latest_version
        and installed_version != "unknown"
    )

    if installed_version == "unknown":
        message = (
            f"Package information for "
            f"{package_name} is unavailable."
        )
    elif up_to_date:
        message = (
            f"{package_name} is up to date."
        )
    else:
        message = (
            f"{package_name} can be updated from "
            f"{installed_version} to {latest_version}."
        )

    return PackageVersionResult(
        package=package_name,
        installed_version=installed_version,
        latest_version=latest_version,
        up_to_date=up_to_date,
        message=message,
    )