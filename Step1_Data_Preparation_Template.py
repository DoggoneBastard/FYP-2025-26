
# ============================================================
# STEP 1: DATA PREPARATION AND CLEANING
# ============================================================
# This script demonstrates how to prepare cryopreservation data 
# from Excel for machine learning algorithms

import pandas as pd
import numpy as np
import re

# Function to extract numeric cooling rate from text
def extract_cooling_rate(cooling_str):
    """Extract primary cooling rate from cooling rate string"""
    if pd.isna(cooling_str):
        return np.nan

    # Look for patterns like "1°C/min", "-1/min", etc.
    match = re.search(r'[-]?(\d+\.?\d*)\s*[°]?C?/min', str(cooling_str), re.IGNORECASE)
    if match:
        return abs(float(match.group(1)))

    # Check for "Directly at -80C" or similar (fast cooling)
    if 'directly' in str(cooling_str).lower():
        return 100.0

    return np.nan

# Function to extract numeric viability
def extract_viability(viability_str):
    """Extract numeric viability value from text"""
    if pd.isna(viability_str):
        return np.nan

    viability_str = str(viability_str)

    # If already numeric, return
    try:
        val = float(viability_str)
        # If value > 1, assume it's percentage, convert to decimal
        if val > 1:
            return val / 100
        return val
    except:
        pass

    # Look for main percentage value (before ±)
    match = re.search(r'(\d+\.?\d*)\s*±?\s*[\d.]*\s*%', viability_str)
    if match:
        return float(match.group(1)) / 100

    return np.nan

# Load the Excel file
excel_file = 'Cryopreservative-Data-Oct.27.xlsx'
df_msc = pd.read_excel(excel_file, sheet_name='MSC')

# Apply cleaning functions
df_msc['Viability_Numeric'] = df_msc['Viability'].apply(extract_viability)
df_msc['Cooling_Rate_Numeric'] = df_msc['Cooling rate'].apply(extract_cooling_rate)
df_msc['DMSO_Numeric'] = pd.to_numeric(df_msc['DMSO usage'], errors='coerce')

# Filter rows with valid viability data
df_clean = df_msc[df_msc['Viability_Numeric'].notna()].copy()

# For missing cooling rates, fill with median
df_clean['Cooling_Rate_Numeric'].fillna(df_clean['Cooling_Rate_Numeric'].median(), inplace=True)

# For missing DMSO, fill with 0 (DMSO-free)
df_clean['DMSO_Numeric'].fillna(0, inplace=True)

# Save cleaned data
df_clean.to_csv('Cleaned_Cryopreservation_Data.csv', index=False)
print(f"Cleaned data saved! Shape: {df_clean.shape}")