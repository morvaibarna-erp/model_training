import pandas as pd

# Specify the input Excel file and output TXT file
input_excel_file = './values.xlsx'
output_txt_file = './values.txt'



# Columns to extract (by name or index)
columns_to_extract = ['Fáljnév                                                  ', 'Mérőállás']  # Ensure these are the correct column names without extra spaces

# Open the Excel file
with pd.ExcelFile(input_excel_file) as xls:
    # Loop through the sheets in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the specified sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Extract the required columns
        extracted_data = df[columns_to_extract]

        # Remove any whitespace from the 'Mérőállás' column
        extracted_data['Mérőállás'] = extracted_data['Mérőállás'].str.replace(r'\s+', '', regex=True)

        extracted_data['Fáljnév                                                  '] = extracted_data['Fáljnév                                                  ']+'.jpg'

        # Append the extracted data to the TXT file without header
        extracted_data.to_csv(output_txt_file, sep='\t', mode='a', index=False, header=False)

print(f"Excel file '{input_excel_file}' has been processed and exported to TXT file '{output_txt_file}' with tab-separated columns.")