import pandas as pd
import re

def clean_first_line(line):
    """
    Clean the first line by removing text and fixing decimal numbers.
    
    Rules:
    1. If the first line has both numbers and text, use regex to remove the text
    2. If a number starts with text followed by decimal, remove text and add 0 before decimal if needed
    """
    # First, try to find all decimal numbers in the line
    # Pattern to match decimal numbers (including those that might start with text)
    decimal_pattern = r'(\d*\.\d+|\d+)'
    numbers = re.findall(decimal_pattern, line)
    
    # If we found at least 3 numbers, use them
    if len(numbers) >= 3:
        # Take the first 3 numbers and convert to float
        try:
            return [float(num) for num in numbers[:3]]
        except ValueError:
            pass
    
    # If the above didn't work, try a more aggressive approach
    # Look for patterns like "text.123" or ".123" and fix them
    # Pattern to match text followed by decimal number
    text_decimal_pattern = r'[a-zA-Z]+\.(\d+)'
    matches = re.findall(text_decimal_pattern, line)
    
    if matches:
        # If we found text-decimal patterns, fix them
        fixed_line = line
        for match in matches:
            # Replace "text.123" with "0.123"
            fixed_line = re.sub(r'[a-zA-Z]+\.' + re.escape(match), '0.' + match, fixed_line)
        
        # Now try to extract numbers again
        numbers = re.findall(decimal_pattern, fixed_line)
        if len(numbers) >= 3:
            try:
                return [float(num) for num in numbers[:3]]
            except ValueError:
                pass
    
    # If still no luck, try to extract any sequence of 3+ numbers
    # This handles cases like "is 4.2 4.2 6.7" where we need to skip "is"
    parts = line.split()
    numeric_parts = []
    for part in parts:
        try:
            val = float(part)
            numeric_parts.append(val)
            if len(numeric_parts) == 3:
                break
        except ValueError:
            continue
    
    return numeric_parts if len(numeric_parts) >= 3 else []

def parse_fn(gen_str):
    """
    Parse generated structure string to extract lengths, angles, species, and coordinates.
    
    The function handles cases where:
    1. The structure may not start from the first line
    2. There may be text before the actual structure data
    3. The first line should contain three numbers (lengths)
    4. The second line should contain three numbers (angles)a
    5. Then pairs of lines: element followed by three fractional coordinates
    """
    # Remove quotes and clean the string
    gen_str = gen_str.strip().strip('"')
    lines = [x.strip() for x in gen_str.split("\n") if len(x.strip()) > 0]
    
    # Find the start of the actual structure data
    # Look for a line that contains three numbers (lengths)
    start_idx = -1
    for i, line in enumerate(lines):
        # Use the cleaning function to extract numbers from the line
        numeric_parts = clean_first_line(line)
        
        # Check if we found 3 positive numbers (likely lengths)
        if len(numeric_parts) == 3 and all(x > 0 for x in numeric_parts):
            start_idx = i
            break
    
    if start_idx == -1 or start_idx >= len(lines) - 1:
        # If we can't find a valid start, return empty structure
        return [], [], [], []
    
    try:
        # Extract lengths (first line after start)
        lengths_line = lines[start_idx]
        # Use the cleaning function to extract numbers
        lengths = clean_first_line(lengths_line)
        
        # Extract angles (second line after start)
        if start_idx + 1 < len(lines):
            angles_line = lines[start_idx + 1]
            # Use the cleaning function to extract numbers
            angles = clean_first_line(angles_line)
        else:
            angles = []
        
        # Extract species and coordinates (pairs starting from third line)
        species = []
        coords = []
        
        # Start from the third line after the start index
        data_start = start_idx + 2
        
        for i in range(data_start, len(lines), 2):
            if i < len(lines):
                # Element line
                element_line = lines[i].strip()
                # Clean the element (remove any non-alphabetic characters at the end)
                element = re.sub(r'[^A-Za-z]', '', element_line)
                if element:  # Only add if we have a valid element
                    species.append(element)
                else:
                    # If no valid element, skip this pair
                    continue
                
                # Coordinates line (next line)
                if i + 1 < len(lines):
                    coords_line = lines[i + 1].strip()
                    try:
                        coord_parts = coords_line.split()
                        if len(coord_parts) >= 3:
                            coord_values = [float(x) for x in coord_parts[:3]]
                            coords.append(coord_values)
                        else:
                            # If coordinates are malformed, add empty coordinates
                            coords.append([0.0, 0.0, 0.0])
                    except ValueError:
                        # If coordinates can't be parsed, add empty coordinates
                        coords.append([0.0, 0.0, 0.0])
                else:
                    # If no coordinates line, add empty coordinates
                    coords.append([0.0, 0.0, 0.0])
        
        return lengths, angles, species, coords
        
    except (ValueError, IndexError) as e:
        # If parsing fails, return empty structure
        return [], [], [], []

def process_csv_file(input_file, output_file):
    """Process a CSV file and save parsed results."""
    print(f"Processing {input_file}...")
    df = pd.read_csv(input_file)
    
    outputs = []
    failed_parses = []
    successful_parses = 0
    
    for idx, gen_str in enumerate(df['gen_str']):
        if pd.isna(gen_str):
            outputs.append({
                'lengths': [],
                'angles': [],
                'species': [],
                'coords': []
            })
            failed_parses.append({
                'index': idx,
                'original_gen_str': gen_str,
                'reason': 'NaN input'
            })
            continue
            
        lengths, angles, species, coords = parse_fn(str(gen_str))
        
        # Check if we got valid data
        if len(lengths) == 3 and len(angles) == 3 and len(species) > 0:
            successful_parses += 1
        else:
            # Record failed parse
            failed_parses.append({
                'index': idx,
                'original_gen_str': str(gen_str),
                'reason': f'Invalid structure: lengths={len(lengths)}, angles={len(angles)}, species={len(species)}'
            })
        
        outputs.append({
            'lengths': lengths,
            'angles': angles,
            'species': species,
            'coords': coords
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} structures...")
    
    result_df = pd.DataFrame(outputs)
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(outputs)} parsed structures to {output_file}")
    print(f"Successfully parsed {successful_parses} out of {len(outputs)} structures")
    
    # Save failed parses
    if failed_parses:
        failed_df = pd.DataFrame(failed_parses)
        failed_output_file = output_file.replace('.csv', '_failed.csv')
        failed_df.to_csv(failed_output_file, index=False)
        print(f"Saved {len(failed_parses)} failed structures to {failed_output_file}")
    
    return result_df

# Process both datasets
if __name__ == "__main__":
    # Process llamat2 dataset
    try:
        process_csv_file("llamat2_9046.csv", "llamat2_9046_parsed.csv")
    except FileNotFoundError:
        print("llamat2_9046.csv not found, skipping...")
    
    # Process llamat3 dataset
    try:
        process_csv_file("llamat3_9046.csv", "llamat3_9046_parsed.csv")
    except FileNotFoundError:
        print("llamat3_9046.csv not found, skipping...")