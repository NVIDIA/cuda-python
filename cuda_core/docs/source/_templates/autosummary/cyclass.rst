.. SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
{% for item in methods %}
.. automethod:: {{ objname }}.{{ item }}

{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
{% for item in attributes %}
.. autoattribute:: {{ objname }}.{{ item }}

{%- endfor %}
{% endif %}
{% endblock %}
