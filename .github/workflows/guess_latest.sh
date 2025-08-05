#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# URL to search
URL="https://developer.download.nvidia.com/compute/cuda/redist/"

# Ensure exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <CUDA_major_version>"
    exit 1
fi

# Accept major version as the first argument
MAJOR_VERSION="$1"

# Fetch the directory listing and extract the latest version number
get_latest_version() {
    # Get the HTML content of the page
    local html_content=$(wget -q -O - "$URL")

    # Extract links matching the pattern redistrib_?.?.?.json
    local files=$(echo "$html_content" | grep -oP "redistrib_${MAJOR_VERSION}\.[0-9]+\.[0-9]+\.json" | cut -d'"' -f2)

    # If files were found, extract the version numbers and find the latest
    if [ -n "$files" ]; then
        # Extract just the version numbers using regex
        local versions=$(echo "$files" | grep -oP "redistrib_\K${MAJOR_VERSION}\.[0-9]+\.[0-9]+(?=\.json)")

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
echo $latest_version
