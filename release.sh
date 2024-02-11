#!/bin/bash
set -e

# Assuming package is already built
# (run build.sh to build the package)

current_version=$(python pyproject.toml --version)
IFS='.' read -ra VERSION <<< "$current_version"
((VERSION[2]++))
new_version="${VERSION[0]}.${VERSION[1]}.${VERSION[2]}"
echo "Current version: $current_version"
echo "New version: $new_version"

# Update the version in pyproject.toml
sed -i "/version='$new_version'/" pyproject.toml


# we also have a build script that will be run independenly (and before this script)

# In our project, we have a branch protection rule which prohibits
# pushing directly to main

# we plan to use this file in github actions

# Commit the version bump
#git commit -am "Bump version to $new_version"
#git tag -a "v$new_version" -m "Version $new_version"
#git push origin main --tags

# Publish to Test PyPI - replace with upload for real PyPI
twine upload --repository testpypi dist/* --api-token $PYPI_API_TOKEN
