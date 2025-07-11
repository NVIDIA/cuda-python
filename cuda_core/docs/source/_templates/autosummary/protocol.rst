.. SPDX-License-Identifier: Apache-2.0

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoprotocol:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoproperty:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
