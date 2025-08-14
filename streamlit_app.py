import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import zipfile
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.graphics.barcode import code128
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Global Constants & Configuration ---
LOCATION_ORDER = ["Heat", "Powder", "Tower", "M", "Production", "Warehouse"]
FONT_NORMAL = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
QTY_TOWER = 0.5
QTY_M = 10

LOCATION_RULES = {
    'POWDER_PREFIXES': ['WB'],
    'TOWER_PATTERNS': [r'^P\d+'],
    'Z_POWDER_PREFIXES': ['Z']
}


def natural_sort_key(s):
    """Returns a tuple for natural sorting (hashable)."""
    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s)))


def format_output_df(df_priority):
    """Takes a DataFrame and formats it, keeping helper columns."""
    if df_priority.empty:
        return pd.DataFrame()
    output_columns = ["Location Type", "Location Description", "Description", "Component", "Batch Nr.1",
                      "Available Quantity", "Quantity required", "Expiry Status", "Needs Highlighting", "Location Priority"]
    df_final = df_priority[output_columns].copy()
    df_final.rename(columns={"Location Type": "Location", "Description": "RM name",
                    "Component": "RM code", "Batch Nr.1": "Batch number"}, inplace=True)
    df_final['RM name'] = df_final['RM name'].str[:20]

    formatted_rows = []
    for location in LOCATION_ORDER:
        subset = df_final[df_final["Location"] == location]
        if not subset.empty:
            header_row = {"Location": location, "Location Description": "", "RM name": "", "RM code": "", "Batch number": "",
                          "Available Quantity": "", "Quantity required": "", "Expiry Status": "", "Needs Highlighting": False, "Location Priority": 0}
            formatted_rows.append(header_row)
            if location == "M":
                subset['sort_key'] = subset['Location Description'].apply(
                    natural_sort_key)
                subset = subset.sort_values(
                    by=['sort_key', 'RM code', 'Batch number']).drop(columns=['sort_key'])
            elif location in ["Tower", "Powder"]:
                subset = subset.sort_values(
                    by=['Location Priority', 'Location Description', 'RM code', 'Batch number'])
            else:
                subset = subset.sort_values(by=['RM name', 'Batch number'])
            for _, row in subset.iterrows():
                formatted_rows.append(row.to_dict())
    if not formatted_rows:
        return pd.DataFrame()
    df_output = pd.DataFrame(formatted_rows)
    df_output.loc[df_output['Location'] == 'Tower', 'Location Description'] = df_output.loc[df_output['Location']
                                                                                            == 'Tower', 'Location Description'].str.replace('TOWER ', '')
    return df_output


def classify_location(desc, qty_required):
    """Classifies a location into a detailed SubType."""
    desc = str(desc).upper().strip()
    if any(desc.startswith(prefix) for prefix in LOCATION_RULES['Z_POWDER_PREFIXES']):
        return "Powder (Z)" if qty_required < QTY_TOWER else "Powder overweight"
    if any(desc.startswith(prefix) for prefix in LOCATION_RULES['POWDER_PREFIXES']):
        return "Powder (WB)"
    if any(re.match(pattern, desc) for pattern in LOCATION_RULES['TOWER_PATTERNS']):
        return "Tower"
    if "HEAT" in desc:
        return "Heat"
    if "POWDER" in desc:
        return "Powder"
    if "TOWER" in desc and qty_required < QTY_TOWER:
        return "Tower"
    if "TOWER" in desc:
        return "Tower overweight"
    if desc.startswith("M") and qty_required < QTY_M:
        return "M"
    if desc.startswith("M"):
        return "M overweight"
    if not desc.startswith("W"):
        return "Production" if "DEFAULT" not in desc else "Default Production"
    return "Warehouse"


def process_data(df):
    """
    Processes the raw DataFrame, assigning priorities using a robust ranking method.
    """
    first_row = df.iloc[0]
    production_date = pd.to_datetime(
        first_row["Current date marked the beginning"], format='%d%m%Y', errors='coerce')
    product_info = {"Production Ticket Nr": first_row["Production Ticket Nr"], "Wording": first_row["Wording"], "Product Code": first_row["Product Code"],
                    "Quantity Launched": df["Quantity launched Theoretical"].astype(float).max(), "Production Date": production_date.date()}
    unique_materials = df.drop_duplicates(subset=["Component", "Description"])
    product_info["Quantity Produced"] = unique_materials["Quantity required"].astype(
        float).sum()
    product_info["Raw Material Count"] = df["Component"].nunique()

    df_sing = df[df["depot location"] == "SING"].copy()
    df_sing['DLUO_dt'] = pd.to_datetime(
        df_sing['DLUO'], format='%d%m%Y', errors='coerce')

    one_month_later = production_date + pd.DateOffset(months=1)
    conditions = [df_sing['DLUO_dt'] < production_date,
                  (df_sing['DLUO_dt'] >= production_date) & (df_sing['DLUO_dt'] < one_month_later)]
    df_sing['Expiry Status'] = np.select(
        conditions, ['Expired', 'Expiring Soon'], default='OK')

    df_sing["Quantity required"] = pd.to_numeric(
        df_sing["Quantity required"], errors='coerce').fillna(0)
    df_sing["Available Quantity"] = pd.to_numeric(df_sing["Available Quantity"].astype(
        str).str.replace(',', '.'), errors='coerce').fillna(0)

    df_sing["Location SubType"] = df_sing.apply(lambda row: classify_location(
        row["Location Description"], row["Quantity required"]), axis=1)

    def map_subtype_to_main_type(subtype):
        if subtype == 'Powder (Z)':
            return 'Tower'
        if 'Powder' in subtype:
            return 'Powder'
        return subtype
    df_sing["Location Type"] = df_sing["Location SubType"].apply(
        map_subtype_to_main_type)

    location_priority = {"Heat": 1, "Powder (Z)": 2.1, "Powder": 2.2, "Powder (WB)": 2.3, "Tower": 3, "M": 4, "Default Production": 5,
                         "Production": 6, "Warehouse": 7, "Tower overweight": 8, "M overweight": 8, "Powder overweight": 8}
    df_sing["Location Priority"] = df_sing["Location SubType"].map(
        location_priority)

    df_sing.sort_values(
        by=["Component", "Location Priority", "Batch Nr.1"], inplace=True)

    # --- FIXED & ROBUST Priority Assignment Logic ---
    # 1. First, rank every batch within its component group
    df_sing['Rank'] = df_sing.groupby('Component').cumcount() + 1

    # 2. Then, assign priorities based on quantity and rank
    all_priority_groups = []
    for component_code, group in df_sing.groupby("Component"):
        required_qty = group["Quantity required"].iloc[0]
        collected_qty = 0
        group_copy = group.copy()

        # Assign 'First Priority' based on cumulative quantity
        first_priority_indices = []
        for index, row in group_copy.iterrows():
            if collected_qty < required_qty:
                first_priority_indices.append(index)
                collected_qty += row['Available Quantity']
            else:
                break
        group_copy.loc[first_priority_indices,
                       'Assigned Priority'] = 'First Priority'

        # Assign other priorities based on the pre-calculated Rank for leftover items
        remaining_rows = group_copy[group_copy['Assigned Priority'].isnull()].copy(
        )

        def assign_by_rank(rank):
            if rank == 2:
                return 'Second Priority'
            if rank == 3:
                return 'Third Priority'
            return 'Leftovers'

        # Use the original rank to assign priorities to the remaining items
        if not remaining_rows.empty:
            # Find the rank of the first non-"First Priority" item
            start_rank = remaining_rows['Rank'].min()
            # Re-calculate rank relative to the start for correct assignment
            remaining_rows['Priority_Rank'] = remaining_rows['Rank'] - \
                start_rank + 2

            group_copy.loc[remaining_rows.index, 'Assigned Priority'] = remaining_rows['Priority_Rank'].apply(
                assign_by_rank)

        group_copy['Needs Highlighting'] = len(first_priority_indices) > 1
        all_priority_groups.append(group_copy)

    if not all_priority_groups:
        return product_info, {}
    df_with_priorities = pd.concat(all_priority_groups)

    priority_dfs_raw = {p_level: df_with_priorities[df_with_priorities['Assigned Priority'] == p_level] for p_level in [
        'First Priority', 'Second Priority', 'Third Priority', 'Leftovers']}
    priority_dfs_formatted = {name: format_output_df(
        df_raw) for name, df_raw in priority_dfs_raw.items() if not df_raw.empty}
    return product_info, priority_dfs_formatted


def generate_pdf(product_info, priority_dfs, barcode_locations, file_configs, content_to_include):
    """Generates PDF(s) based on file configs and content selection."""
    generated_files = []
    for config in file_configs:
        file_num, total_files, locations_for_this_file = config[
            'file_num'], config['total_files'], config['locations']
        pdf_filename = f"{product_info['Production Ticket Nr']}_{file_num}_of_{total_files}.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=A4, leftMargin=0.5*inch,
                                rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='LeftAlign',
                   alignment=TA_LEFT, fontName=FONT_NORMAL))
        styles['Normal'].fontName = FONT_NORMAL
        styles['h1'].fontName = FONT_BOLD

        elements, content_added_to_this_pdf = [], False

        title_text = f"Production Ticket Information - {product_info['Production Ticket Nr']}"
        if total_files > 1:
            title_text += f" ({file_num} / {total_files})"
        elements.append(Paragraph(title_text, styles['h1']))

        info_copy = {k: v for k, v in product_info.items() if k !=
                     "Production Ticket Nr"}
        info_data = [[key, str(value)] for key, value in info_copy.items()]
        info_table = Table(info_data, colWidths=[2*inch, 5*inch])
        info_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME', (0, 0), (0, -1), FONT_BOLD), ('FONTNAME',
                            (1, 0), (1, -1), FONT_NORMAL), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.2*inch))

        is_first_content_block = True

        # --- UPDATED: Loop through user-selected content only ---
        for priority_name in content_to_include:
            if priority_name in priority_dfs:
                df_output = priority_dfs[priority_name][priority_dfs[priority_name]['Location'].isin(
                    locations_for_this_file)]
                if not df_output.empty:
                    content_added_to_this_pdf, is_first_content_block = True, False if is_first_content_block else elements.append(
                        PageBreak()) or False
                    elements.append(Paragraph(priority_name, styles['h1']))
                    elements.append(Spacer(1, 0.1*inch))
                    headers = ["Location", "Location\nDescription", "RM name", "RM code\n& Barcode",
                               "Batch\nnumber", "Available\nQuantity", "Quantity\nRequired"]
                    table_data = [headers]
                    for _, row in df_output.iterrows():
                        if row['RM code'] == '':
                            table_data.append(
                                list(row.drop(['Expiry Status', 'Needs Highlighting', 'Location Priority'])))
                            continue
                        barcode_cell = Paragraph(
                            str(row['RM code']), styles['Normal'])
                        if row['Location'] in barcode_locations:
                            barcode = code128.Code128(
                                str(row['RM code']), barHeight=0.2*inch, barWidth=0.008*inch)
                            barcode_cell = [barcode_cell,
                                            Spacer(10, 0), barcode]
                        data_row = [row['Location'], Paragraph(str(row['Location Description']), styles['Normal']), Paragraph(str(
                            row['RM name']), styles['LeftAlign']), barcode_cell, Paragraph(str(row['Batch number']), styles['Normal']), row['Available Quantity'], row['Quantity required']]
                        table_data.append(data_row)

                    output_table = Table(table_data, colWidths=[
                                         0.8*inch, 1*inch, 2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch], repeatRows=1)
                    base_style = [('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                  ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD), ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (2, 1), (2, -1), 'LEFT')]
                    dynamic_styles = []
                    for i, row_data in enumerate(table_data[1:], 1):
                        if row_data[1] == '':
                            dynamic_styles.extend([('BACKGROUND', (0, i), (-1, i), colors.lightgrey), (
                                'TEXTCOLOR', (0, i), (-1, i), colors.black), ('FONTNAME', (0, i), (0, i), FONT_BOLD)])
                        else:
                            original_row = df_output[df_output['Batch number'] == str(
                                row_data[4].text)].iloc[0]
                            if original_row['Expiry Status'] == 'Expired':
                                dynamic_styles.append(
                                    ('BACKGROUND', (2, i), (2, i), colors.lightcoral))
                            elif original_row['Expiry Status'] == 'Expiring Soon':
                                dynamic_styles.append(
                                    ('BACKGROUND', (2, i), (2, i), colors.moccasin))
                            if original_row['Needs Highlighting']:
                                dynamic_styles.append(
                                    ('BACKGROUND', (5, i), (6, i), colors.yellow))
                    output_table.setStyle(TableStyle(
                        base_style + dynamic_styles))
                    elements.append(output_table)
        if content_added_to_this_pdf:
            doc.build(elements)
            generated_files.append(pdf_filename)
    return generated_files


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Production Ticket Processor")

# --- UI for PDF Splitting and Content Selection ---
st.sidebar.header("PDF Generation Options")
split_option = st.sidebar.radio(
    "PDF Splitting:", ("Single File", "Split into 2 Files", "Split into 3 Files"))

file_configs, is_valid_config = [], True
if split_option == "Single File":
    file_configs = [
        {'file_num': 1, 'total_files': 1, 'locations': LOCATION_ORDER}]
else:
    num_splits = 2 if "2" in split_option else 3
    assignments = []  # Initialize an empty list first

    # Use a traditional for loop to build the assignments step-by-step
    for i in range(num_splits):
        # Calculate available options based on what's already been assigned
        assigned_so_far = sum(assignments, [])
        available_options = [
            loc for loc in LOCATION_ORDER if loc not in assigned_so_far]

        # Create the multiselect widget for the current file
        selection = st.sidebar.multiselect(
            f"Locations for File {i+1}:",
            available_options
        )
        assignments.append(selection)

    # Now, perform the validation check after the loop is complete
    if len(sum(assignments, [])) != len(LOCATION_ORDER):
        st.sidebar.warning("All locations must be assigned to a file.")
        is_valid_config = False
    else:
        file_configs = [{'file_num': i+1, 'total_files': num_splits,
                         'locations': assignments[i]} for i in range(num_splits)]

st.sidebar.markdown("---")
barcode_locations_selection = st.sidebar.multiselect(
    "Generate barcodes for which locations?", LOCATION_ORDER, default=["Tower"])

# --- NEW: UI for Content Selection ---
st.sidebar.markdown("---")
st.sidebar.write("Select Content to Include:")
include_p2 = st.sidebar.checkbox("Include Second Priority", True)
include_p3 = st.sidebar.checkbox("Include Third Priority", True)
include_leftovers = st.sidebar.checkbox("Include Leftovers", True)

content_to_include = ['First Priority']
if include_p2:
    content_to_include.append('Second Priority')
if include_p3:
    content_to_include.append('Third Priority')
if include_leftovers:
    content_to_include.append('Leftovers')

st.write("Upload an Excel file to see the first-priority picking list.")
uploaded_file = st.file_uploader(
    "Choose an XLS or XLSX file", type=["xls", "xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df_preview = pd.read_excel(xls, sheet_name=sheet_name)
        header_row_index = df_preview.apply(lambda row: row.astype(
            str).str.contains("Batch Nr").any(), axis=1).idxmax() + 1
        string_converters = {'Production Ticket Nr': str, 'Product Code': str, 'Batch Nr': str,
                             'Current date marked the beginning': str, 'Component': str, 'Batch Nr.1': str, 'DLUO': str}
        df = pd.read_excel(xls, sheet_name=sheet_name,
                           header=header_row_index, converters=string_converters)

        product_info, priority_dataframes = process_data(df)
        st.header("Production Information")
        st.json(product_info, expanded=False)
        st.header("First Priority Picking List (Preview)")

        if 'First Priority' in priority_dataframes and not priority_dataframes['First Priority'].empty:
            st.dataframe(priority_dataframes['First Priority'].drop(columns=[
                         'Location', 'Expiry Status', 'Needs Highlighting', 'Location Priority']))
            if is_valid_config and st.sidebar.button("Generate Full Picking PDF"):
                with st.spinner('Creating PDF(s)...'):
                    pdf_filenames = generate_pdf(
                        product_info, priority_dataframes, barcode_locations_selection, file_configs, content_to_include)
                    if not pdf_filenames:
                        st.sidebar.error(
                            "No content available for the selected locations/priorities.")
                    elif len(pdf_filenames) == 1:
                        with open(pdf_filenames[0], "rb") as f:
                            st.sidebar.download_button(
                                "Download PDF", f, file_name=pdf_filenames[0], mime="application/pdf")
                        os.remove(pdf_filenames[0])
                    else:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
                            for f in pdf_filenames:
                                zf.write(f, os.path.basename(f))
                        st.sidebar.download_button("Download Picking Lists (ZIP)", zip_buffer.getvalue(
                        ), f"{product_info['Production Ticket Nr']}_picking_lists.zip", "application/zip")
                        for f in pdf_filenames:
                            os.remove(f)
        else:
            st.warning("No first-priority items were found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(
            "Please check file format and ensure all locations are assigned if splitting.")
