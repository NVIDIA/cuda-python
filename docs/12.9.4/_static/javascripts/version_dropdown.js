// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

function change_current_version(event) {
  event.preventDefault();

  var selectedVersion = event.target.textContent;
  var currentVersion = document.getElementById('currentVersion');

  // need to update both the on-screen state and the internal (persistent) storage
  currentVersion.textContent = selectedVersion;
  sessionStorage.setItem("currentVersion", selectedVersion);

  // Navigate to the clicked URL
  window.location.href = event.target.href;
}


function add_version_dropdown(jsonLoc, targetLoc, currentVersion) {
  var otherVersionsDiv = document.getElementById('otherVersions');

  fetch(jsonLoc)
    .then(function(response) {
      return response.json();
    })
    .then(function(data) {
      var versions = data;

      if (Object.keys(versions).length >= 1) {
        var dlElement = document.createElement('dl');
        var dtElement = document.createElement('dt');
        dtElement.textContent = 'Versions';
        dlElement.appendChild(dtElement);

        for (var ver in versions) {
          var url = versions[ver];
          var ddElement = document.createElement('dd');
          var aElement = document.createElement('a');
          aElement.setAttribute('href', targetLoc + url);
          aElement.textContent = ver;

          if (ver === currentVersion) {
            var strongElement = document.createElement('strong');
            strongElement.appendChild(aElement);
            aElement = strongElement;
          }

          ddElement.appendChild(aElement);
          // Attach event listeners to version links
          ddElement.addEventListener('click', change_current_version);
          dlElement.appendChild(ddElement);
        }

        otherVersionsDiv.innerHTML = '';
        otherVersionsDiv.appendChild(dlElement);
      }
    })
    .catch(function(error) {
      console.error('Error fetching version.json:', error);
    });
}
