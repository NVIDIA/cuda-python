// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

function add_version_dropdown(jsonLoc, currentVersion) {
  var otherVersionsDiv = document.getElementById('otherVersions');

  fetch(jsonLoc)
    .then(function(response) {
      return response.json();
    })
    .then(function(data) {
      var versions = data;

      if (Array.isArray(versions) && versions.length >= 1) {
        var dlElement = document.createElement('dl');
        var dtElement = document.createElement('dt');
        dtElement.textContent = 'Versions';
        dlElement.appendChild(dtElement);

        for (var index = 0; index < versions.length; index++) {
          var ver = versions[index].version;
          var url = versions[index].url;
          var ddElement = document.createElement('dd');
          var aElement = document.createElement('a');
          aElement.setAttribute('href', url);
          aElement.textContent = ver;

          if (ver === currentVersion) {
            var strongElement = document.createElement('strong');
            strongElement.appendChild(aElement);
            aElement = strongElement;
          }

          ddElement.appendChild(aElement);
          dlElement.appendChild(ddElement);
        }

        otherVersionsDiv.innerHTML = '';
        otherVersionsDiv.appendChild(dlElement);
      }
    })
    .catch(function(error) {
      console.error('Error fetching nv-versions.json:', error);
    });
}
