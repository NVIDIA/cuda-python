#!/usr/bin/env python3

import argparse
import json
from jinja2 import Environment, FileSystemLoader
import os
import re

# TODO: make this work for arbitrary context. ie. implement replace_using_context()
def replace_placeholder(source_str, variable_name, variable_value):
    # Escaping any regex special characters in variable_name
    variable_name_escaped = re.escape(variable_name)

    # Using regular expression to replace ${variable_name} with actual variable_value
    # \s* means any amount of whitespace (including none)
    # pattern = rf'\$\{{\s*\{{\s*{variable_name_escaped}\s*\}}\s*\}}'
    pattern = rf'<<\s*{variable_name_escaped}\s*>>'
    return re.sub(pattern, variable_value.strip(), source_str)

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description='Render a Jinja2 template using a JSON context.')
parser.add_argument('template_file', type=str, help='Path to the Jinja2 template file (with .j2 extension).')
parser.add_argument('json_file', type=str, help='Path to the JSON file to use as the rendering context.')
parser.add_argument('output_file', type=str, help='Path to the output file.')

args = parser.parse_args()

# Load JSON file as the rendering context
with open(args.json_file, 'r') as file:
    context = json.load(file)

# Setup Jinja2 environment and load the template
env = Environment(
    loader=FileSystemLoader(searchpath='./'),
    variable_start_string='<<',
    variable_end_string='>>',
    block_start_string='<%',
    block_end_string='%>',
    comment_start_string='<#',
    comment_end_string='#>')
env.filters['replace_placeholder'] = replace_placeholder

template = env.get_template(args.template_file)

# Render the template with the context
rendered_content = template.render(context)
# print(rendered_content)

with open(args.output_file, 'w') as file:
    file.write(rendered_content)

print(f'Template rendered successfully. Output saved to {args.output_file}')
