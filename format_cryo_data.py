import pandas as pd
import numpy as np
import re

def parse_percentage(value):
    """Convert percentage strings to decimal (e.g., '10%' -> 0.10)"""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return value / 100 if value > 1 else value
    
    value = str(value).strip()
    match = re.search(r'(\d+\.?\d*)\s*%', value)
    if match:
        return float(match.group(1)) / 100
    
    try:
        num = float(value)
        return num / 100 if num > 1 else num
    except:
        return np.nan

def extract_concentration(text, ingredient, unit='%'):
    """Extract concentration of specific ingredient from text"""
    if pd.isna(text):
        return np.nan
    
    text = str(text).lower()
    ingredient = ingredient.lower()
    
    # Pattern for percentage
    if unit == '%':
        pattern = rf'(\d+\.?\d*)\s*%?\s*{re.escape(ingredient)}'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)) / 100 if float(match.group(1)) > 1 else float(match.group(1))
    
    # Pattern for molar (M) or millimolar (mM)
    elif unit in ['M', 'mM']:
        pattern = rf'(\d+\.?\d*)\s*m?\s*{re.escape(ingredient)}'
        match = re.search(pattern, text)
        if match:
            val = float(match.group(1))
            # Check if it's M or mM
            if 'm' in text[match.start():match.end()].lower() and 'mm' not in text[match.start():match.end()].lower():
                return val * 1000  # Convert M to mM
            return val
    
    return np.nan

def parse_cooling_rate(text):
    """Extract cooling rate information"""
    if pd.isna(text):
        return {'min_rate': np.nan, 'max_rate': np.nan, 'avg_rate': np.nan, 
                'method': 'unknown', 'controlled': 0}
    
    text = str(text).lower()
    
    # Extract all rates
    rates = re.findall(r'(\d+\.?\d*)\s*°?c\s*/\s*min', text)
    rates = [float(r) for r in rates]
    
    # Determine method
    if 'direct' in text or 'plunge' in text or 'flash' in text:
        method = 'direct_plunge'
        controlled = 0
    elif 'uncontrolled' in text or 'mr frosty' in text or 'coolcell' in text:
        method = 'passive_cooling'
        controlled = 0
    elif 'controlled' in text or len(rates) > 1:
        method = 'controlled_rate'
        controlled = 1
    else:
        method = 'slow_freeze'
        controlled = 0
    
    result = {
        'min_rate': min(rates) if rates else np.nan,
        'max_rate': max(rates) if rates else np.nan,
        'avg_rate': np.mean(rates) if rates else np.nan,
        'method': method,
        'controlled': controlled
    }
    
    return result

def process_cryo_data(input_file, output_file):
    """Main processing function"""
    
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Create new columns for individual ingredients
    ingredients_data = {
        'DMSO_pct': [],
        'Ethylene_Glycol_pct': [],
        'Ethylene_Glycol_mM': [],
        'Propylene_Glycol_pct': [],
        'Glycerol_pct': [],
        'Trehalose_mM': [],
        'Sucrose_mM': [],
        'Glucose_mM': [],
        'Mannitol_mM': [],
        'Raffinose_mM': [],
        'FBS_pct': [],
        'FCS_pct': [],
        'Human_Serum_pct': [],
        'HSA_pct': [],
        'BSA_pct': [],
        'Albumin_pct': [],
        'PVP_pct': [],
        'PEG_pct': [],
        'HES_pct': [],
        'Dextran_pct': [],
        'Ficoll_pct': [],
        'Taurine_mM': [],
        'Ectoine_pct': [],
        'Proline_pct': [],
        'Isoleucine_mM': [],
        'Creatine_mM': [],
        'Methylcellulose_pct': [],
        'ROCKi_uM': [],
        'S1P_uM': [],
        'Polyampholyte_mgml': [],
        'COOH_PLL_pct': [],
        'Sericin_pct': [],
        'Betaine_pct': [],
    }
    
    # Cooling rate columns
    cooling_data = {
        'cooling_min_rate_C_per_min': [],
        'cooling_max_rate_C_per_min': [],
        'cooling_avg_rate_C_per_min': [],
        'cooling_method': [],
        'cooling_controlled': []
    }
    
    # Process each row
    for idx, row in df.iterrows():
        ingredients_text = row['All ingredients in cryoprotective solution']
        
        # Extract DMSO
        dmso_usage = row.get('DMSO usage', np.nan)
        dmso_pct = parse_percentage(dmso_usage)
        if pd.isna(dmso_pct):
            dmso_pct = extract_concentration(ingredients_text, 'dmso', '%')
        ingredients_data['DMSO_pct'].append(dmso_pct)
        
        # Extract other CPAs
        ingredients_data['Ethylene_Glycol_pct'].append(
            extract_concentration(ingredients_text, 'ethylene glycol', '%') or
            extract_concentration(ingredients_text, 'eg', '%')
        )
        ingredients_data['Ethylene_Glycol_mM'].append(
            extract_concentration(ingredients_text, 'ethylene glycol', 'mM') or
            extract_concentration(ingredients_text, 'eg', 'mM')
        )
        ingredients_data['Propylene_Glycol_pct'].append(
            extract_concentration(ingredients_text, 'propylene glycol', '%') or
            extract_concentration(ingredients_text, 'proh', '%')
        )
        ingredients_data['Glycerol_pct'].append(
            extract_concentration(ingredients_text, 'glycerol', '%')
        )
        
        # Sugars
        ingredients_data['Trehalose_mM'].append(
            extract_concentration(ingredients_text, 'trehalose', 'mM')
        )
        ingredients_data['Sucrose_mM'].append(
            extract_concentration(ingredients_text, 'sucrose', 'mM')
        )
        ingredients_data['Glucose_mM'].append(
            extract_concentration(ingredients_text, 'glucose', 'mM')
        )
        ingredients_data['Mannitol_mM'].append(
            extract_concentration(ingredients_text, 'mannitol', 'mM')
        )
        ingredients_data['Raffinose_mM'].append(
            extract_concentration(ingredients_text, 'raffinose', 'mM')
        )
        
        # Proteins/Sera
        ingredients_data['FBS_pct'].append(extract_concentration(ingredients_text, 'fbs', '%'))
        ingredients_data['FCS_pct'].append(extract_concentration(ingredients_text, 'fcs', '%'))
        ingredients_data['Human_Serum_pct'].append(
            extract_concentration(ingredients_text, 'human serum', '%') or
            extract_concentration(ingredients_text, 'hs', '%')
        )
        ingredients_data['HSA_pct'].append(
            extract_concentration(ingredients_text, 'human serum albumin', '%') or
            extract_concentration(ingredients_text, 'hsa', '%') or
            extract_concentration(ingredients_text, 'human albumin', '%')
        )
        ingredients_data['BSA_pct'].append(extract_concentration(ingredients_text, 'bsa', '%'))
        ingredients_data['Albumin_pct'].append(extract_concentration(ingredients_text, 'albumin', '%'))
        
        # Polymers
        ingredients_data['PVP_pct'].append(extract_concentration(ingredients_text, 'pvp', '%'))
        ingredients_data['PEG_pct'].append(extract_concentration(ingredients_text, 'peg', '%'))
        ingredients_data['HES_pct'].append(
            extract_concentration(ingredients_text, 'hes', '%') or
            extract_concentration(ingredients_text, 'hydroxyethyl starch', '%')
        )
        ingredients_data['Dextran_pct'].append(extract_concentration(ingredients_text, 'dextran', '%'))
        ingredients_data['Ficoll_pct'].append(extract_concentration(ingredients_text, 'ficoll', '%'))
        
        # Amino acids and other compounds
        ingredients_data['Taurine_mM'].append(extract_concentration(ingredients_text, 'taurine', 'mM'))
        ingredients_data['Ectoine_pct'].append(extract_concentration(ingredients_text, 'ectoine', '%'))
        ingredients_data['Proline_pct'].append(extract_concentration(ingredients_text, 'proline', '%'))
        ingredients_data['Isoleucine_mM'].append(extract_concentration(ingredients_text, 'isoleucine', 'mM'))
        ingredients_data['Creatine_mM'].append(extract_concentration(ingredients_text, 'creatine', 'mM'))
        ingredients_data['Methylcellulose_pct'].append(extract_concentration(ingredients_text, 'methylcellulose', '%'))
        
        # Additives
        rock_match = re.search(r'(\d+\.?\d*)\s*[uµ]m\s*rock', str(ingredients_text).lower())
        ingredients_data['ROCKi_uM'].append(float(rock_match.group(1)) if rock_match else np.nan)
        
        s1p_match = re.search(r'(\d+\.?\d*)\s*[uµ]m\s*s1p', str(ingredients_text).lower())
        ingredients_data['S1P_uM'].append(float(s1p_match.group(1)) if s1p_match else np.nan)
        
        poly_match = re.search(r'(\d+\.?\d*)\s*mg/ml\s*polyampholyte', str(ingredients_text).lower())
        ingredients_data['Polyampholyte_mgml'].append(float(poly_match.group(1)) if poly_match else np.nan)
        
        cooh_match = re.search(r'(\d+\.?\d*)\s*%.*cooh-pll', str(ingredients_text).lower())
        ingredients_data['COOH_PLL_pct'].append(float(cooh_match.group(1))/100 if cooh_match else np.nan)
        
        ingredients_data['Sericin_pct'].append(extract_concentration(ingredients_text, 'sericin', '%'))
        ingredients_data['Betaine_pct'].append(extract_concentration(ingredients_text, 'betaine', '%'))
        
        # Parse cooling rate
        cooling_info = parse_cooling_rate(row.get('Cooling rate', ''))
        cooling_data['cooling_min_rate_C_per_min'].append(cooling_info['min_rate'])
        cooling_data['cooling_max_rate_C_per_min'].append(cooling_info['max_rate'])
        cooling_data['cooling_avg_rate_C_per_min'].append(cooling_info['avg_rate'])
        cooling_data['cooling_method'].append(cooling_info['method'])
        cooling_data['cooling_controlled'].append(cooling_info['controlled'])
    
    # Add new columns to dataframe
    for col, data in ingredients_data.items():
        df[col] = data
    
    for col, data in cooling_data.items():
        df[col] = data
    
    # Parse viability and recovery
    df['Viability_numeric'] = df['Viability'].apply(parse_percentage)
    df['Recovery_numeric'] = df['Recovery'].apply(parse_percentage)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Processed {len(df)} rows")
    print(f"✓ Created {len(ingredients_data) + len(cooling_data)} new ingredient/cooling columns")
    print(f"✓ Saved to: {output_file}")
    
    # Print summary statistics
    print("\n=== Ingredient Summary ===")
    for col in sorted(ingredients_data.keys()):
        non_zero = df[col].notna().sum()
        if non_zero > 0:
            print(f"{col}: {non_zero} formulations contain this ingredient")

if __name__ == "__main__":
    input_file = "Cryopreservative Data Nov.12.csv"  # Your input file
    output_file = "cryopreservative_ml_ready.csv"
    
    process_cryo_data(input_file, output_file)