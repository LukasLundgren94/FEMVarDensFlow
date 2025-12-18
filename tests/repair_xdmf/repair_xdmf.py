# Copyright (c) 2023 Lukas Lundgren
# Licensed under the MIT License – see the LICENSE file for details

import re
import sys
# This is used to fix a bug in the XDMF writer. The bug occurs by installing fenics using spack and this scripts fixes it.
def modify_xdmf(file_path):
    # Read the original content from the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Template for replacements, using placeholders for the number
    replacement_templates = [
        'h5:/Mesh/{}/mesh/topology</DataItem>',
        'h5:/Mesh/{}/mesh/geometry</DataItem>',
        'h5:/VisualisationVector/{}</DataItem>'
    ]

    # Find all instances to be replaced
    instances = re.findall(r'h5</DataItem>', content)

    # Replace each instance in a cyclic (modulo) way
    for i, instance in enumerate(instances):
        cycle_index = i // len(replacement_templates)  # Determines the cycle number
        replacement = replacement_templates[i % len(replacement_templates)].format(cycle_index)
        content = re.sub(r'h5</DataItem>', replacement, content, count=1)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 script.py filename.xdmf")
    else:
        for file_path in sys.argv[1:]:
            print(f"Processing file: {file_path}")
            modify_xdmf(file_path)
            print(f"File '{file_path}' has been modified.")