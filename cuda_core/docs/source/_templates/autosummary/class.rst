.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
{% for item in methods %}
{{ objname }}.{{ item }}
{{ "-" * (objname|length + 1 + item|length) }}

.. automethod:: {{ item }}

{% endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
{% for item in attributes %}
{{ objname }}.{{ item }}
{{ "-" * (objname|length + 1 + item|length) }}

.. autoproperty:: {{ item }}

{% endfor %}
{% endif %}
{% endblock %}
