from lxml        import etree
from collections import defaultdict

def run(filename):

    # Parse the XDMF file with lxml
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(filename, parser)
    root = tree.getroot()

    # Define namespace and get the xi:include element that serves as a pointer to the mesh
    ns = {'xi': 'https://www.w3.org/2001/XInclude'}
    xi_include_element = root.xpath("//xi:include", namespaces=ns)
    if xi_include_element: xi_include_element = xi_include_element[0]  # Use the first found xi:include element

    # Extract the mesh Grid
    mesh_grid = root.xpath("//Grid[@Name='mesh']")

    # Extract and group Grid elements by Time value
    time_groups = defaultdict(list)
    for collection_grid in root.xpath("//Grid[@GridType='Collection']"):
        grids = collection_grid.xpath(".//Grid[@GridType='Uniform']")

        # Group Grid elements by Time value
        for grid in grids:
            time_value = grid.find("Time").attrib['Value']
            time_groups[time_value].append(grid)

    # Create new Grid elements with grouped Time values
    new_collection_grids = []
    for time_value, grids in time_groups.items():
        # Create a new Grid element
        new_grid = etree.Element("Grid", Name=f"merged_time_{time_value}", GridType="Uniform")

        # Add xi:include element. Use a deep copy to avoid reusing the same element
        new_grid.append(etree.Element("{https://www.w3.org/2001/XInclude}include", attrib=xi_include_element.attrib))

        # Add Time element
        time_element = etree.Element("Time", Value=time_value)
        new_grid.append(time_element)

        # Add all Attributes from the grids in this group
        for grid in grids:
            for attribute in grid.findall("Attribute"):
                new_grid.append(attribute)

        # Create a new Collection Grid and add the merged Grid to it
        new_collection_grid = etree.Element("Grid", Name=f"Collection_{time_value}", GridType="Collection", CollectionType="Temporal")
        new_collection_grid.append(new_grid)
        new_collection_grids.append(new_collection_grid)

    # Clear original collection grids and append the new ones
    root.clear()
    root.append(etree.Element("Domain"))
    for child in root:
        child.append(mesh_grid[0])
        for new_collection_grid in new_collection_grids:
            child.append(new_collection_grid)

    # Write the modified XDMF back to file
    with open(filename, 'wb') as f:
        tree.xpath("//Xdmf")[0].set("Version", "3.0") # Add version to XDMF header
        f.write(etree.tostring(tree, pretty_print=True, xml_declaration=True, encoding='UTF-8'))