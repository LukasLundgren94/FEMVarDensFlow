# Copyright (c) 2023 Lukas Lundgren
# Licensed under the MIT License – see the LICENSE file for details

import re
import sys
# This is used to fix a bug in the XDMF writer. The bug occurs by installing fenics using spack and this scripts fixes it.

def extract_name_from_xdmf(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to find the 'Name' attribute within the <Grid> tag
    match = re.search(r'<Grid[^>]*Name="([^"]*)"', content)
    if match:
        return match.group(1)
    else:
        return None

# Replace 'your_file.xdmf' with the path to your actual XDMF file
# file_path = "S900.xdmf"

# if name_attribute:
    # print(f"The 'Name' attribute in the <Grid> element is: {name_attribute}")
# else:
    # print("The 'Name' attribute was not found in the <Grid> element.")

def modify_xdmf(file_path):
    name_attribute = extract_name_from_xdmf(file_path)
    name_attribute_0 = name_attribute + "_0"

    # Read the original content from the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Template for replacements, using placeholders for the number
    replacement_templates = [
        'h5:{0}/{1}/Mesh/mesh/topology</DataItem>',
        'h5:{0}/{1}/Mesh/mesh/geometry</DataItem>',
        'h5:{0}/{1}/cell_dofs</DataItem>',
        'h5:{0}/{1}/vector</DataItem>',
        'h5:{0}/{1}/x_cell_dofs</DataItem>',
        'h5:{0}/{1}/cells</DataItem>'
    ]

    # Find all instances to be replaced
    instances = re.findall(r'h5</DataItem>', content)

    # Replace each instance in a cyclic (modulo) way
    for i, instance in enumerate(instances):
        cycle_index = i // len(replacement_templates)  # Determines the cycle number
        replacement = replacement_templates[i % len(replacement_templates)].format(name_attribute,name_attribute_0)
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