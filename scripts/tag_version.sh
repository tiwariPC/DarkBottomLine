#!/bin/bash
# Tag current HEAD (last feature commit), update _version.py, push tag + GitHub release.
set -e

DATE=$(date +%Y.%m.%d)
HASH=$(git rev-parse --short HEAD)
VERSION="${DATE}-${HASH}"

echo "Version: $VERSION"

# Tag current HEAD (last feature commit) BEFORE version bump commit
git tag "$VERSION"
echo "Tagged HEAD as: $VERSION"

# Update _version.py
VERSION_FILE="$(dirname "$0")/../darkbottomline/_version.py"
cat > "$VERSION_FILE" <<EOF
"""Package version information for DarkBottomLine."""

__version__ = "${VERSION}"
EOF

echo "Updated _version.py"

# Commit version bump (after tag — tag stays on feature commit)
git add "$VERSION_FILE"
git commit -m "chore(version): bump to ${VERSION}"

# Push commit + tag + GitHub release
read -rp "Push to origin and create GitHub release? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    git push origin HEAD
    git push origin "$VERSION"
    gh release create "$VERSION" \
        --title "$VERSION" \
        --notes "Release ${VERSION}" \
        --target "$(git rev-parse $VERSION)"
    echo "GitHub release created: $VERSION"
else
    echo "Skipped. Run manually:"
    echo "  git push origin HEAD && git push origin $VERSION"
    echo "  gh release create $VERSION --title $VERSION --notes 'Release $VERSION'"
fi
