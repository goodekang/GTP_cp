import base64
import re
import os

def update_xml_with_images():
    xml_path = "results/figures/fig1_concept/fig1_concept_drawio.xml"
    
    # Map IDs to the generated image files
    # ID 70: Original Network
    # ID 71: Perturbed Network
    # ID 73: KM Plot
    replacements = {
        "70": "results/figures/elements/network_original.png",
        "71": "results/figures/elements/network_perturbed.png",
        "73": "results/figures/elements/km_plot_outcome.png"
    }
    
    print(f"Reading XML: {xml_path}")
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    for xml_id, img_path in replacements.items():
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            continue
            
        print(f"Processing ID {xml_id} with {img_path}...")
        
        # Read and encode image
        with open(img_path, "rb") as img_f:
            b64_str = base64.b64encode(img_f.read()).decode("utf-8")
            
        # Define the new style for the image
        # aspect=fixed to preserve ratio
        # shape=image is the key
        # noLabel=1 hides any text value
        new_style = (
            f"shape=image;html=1;verticalAlign=top;aspect=fixed;imageAspect=0;"
            f"image=data:image/png;base64,{b64_str};"
            "strokeColor=none;fillColor=none;" 
        )
        
        # Regex to target the specific mxCell line
        # Matches: <mxCell id="70" ... value="ANYTHING" ... style="ANYTHING"
        # We handle the variability of attribute order by just looking for the tag start and the specific ID
        
        # Strategy: Find the start of the tag, then replace value and style attributes inside it.
        # But attributes can be in any order. 
        # Simpler approach: Locate the specific string segment we saw in grep and replace it precisely if possible.
        # The grep showed: <mxCell id="70" value="PLACEHOLDER..." style="..." ...
        
        # Let's try to construct a regex that is flexible enough.
        # We look for <mxCell id="{xml_id}" followed by anything until >
        
        pattern = re.compile(rf'(<mxCell\s+id="{xml_id}"\s+[^>]*>)')
        
        match = pattern.search(content)
        if match:
            tag_content = match.group(1)
            print(f"Found tag for ID {xml_id}")
            
            # Replace 'value="..."' with 'value=""'
            new_tag = re.sub(r'value="[^"]*"', 'value=""', tag_content)
            
            # Replace 'style="..."' with new style
            # If style exists, replace it. If not, append it (unlikely here).
            if 'style="' in new_tag:
                 new_tag = re.sub(r'style="[^"]*"', f'style="{new_style}"', new_tag)
            else:
                # Insert style before the closing > (simplified)
                new_tag = new_tag.replace('>', f' style="{new_style}">')
                
            # Perform the substitution in the main content
            content = content.replace(tag_content, new_tag)
            print(f"Updated ID {xml_id}")
        else:
            print(f"Warning: Could not find mxCell with id={xml_id}")

    print("Writing updated XML...")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Done.")

if __name__ == "__main__":
    update_xml_with_images()







