#!/bin/bash
# URL to search
URL="https://developer.download.nvidia.com/compute/cuda/redist/"

# Fetch the directory listing and extract the latest version number
get_latest_version() {
    # Get the HTML content of the page
    local html_content=$(wget -q -O - "$URL")

    # Extract links matching the pattern redistrib_?.?.?.json
    local files=$(echo "$html_content" | grep -oP 'redistrib_[0-9]+\.[0-9]+\.[0-9]+\.json' | cut -d'"' -f2)

    # If files were found, extract the version numbers and find the latest
    if [ -n "$files" ]; then
        # Extract just the version numbers using regex
        local versions=$(echo "$files" | grep -oP 'redistrib_\K[0-9]+\.[0-9]+\.[0-9]+(?=\.json)')

        # Sort the versions and get the latest
        local latest_version=$(echo "$versions" | sort -V | tail -n 1)
        echo "$latest_version"
    else
        echo "No files matching the pattern were found."
        return 1
    fi
}

# Call the function and store the result
latest_version=$(get_latest_version)
