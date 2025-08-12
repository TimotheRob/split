import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import os
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.graphics.barcode import code128
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# --- Global Constants ---
# Define constants here for easy modification
LOCATION_ORDER = ["Heat", "Powder", "Tower", "M", "Production", "Warehouse"]
FONT_NORMAL = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
QTY_TOWER = .5
QTY_M = 10


def format_output_df(df_priority):
    """
    Takes a DataFrame for a specific priority level and formats it for display or PDF export.
    This includes renaming columns, sorting, truncating text, and adding category headers.
    """
    if df_priority.empty:
        return pd.DataFrame()

    # Step 1: Organize output table, including new status columns
    output_columns = [
        "Location Type", "Location Description", "Description", "Component", "Batch Nr.1",
        "Available Quantity", "Quantity required", "Expiry Status", "Needs Highlighting"
    ]
    df_final = df_priority[output_columns].copy()
    df_final.rename(columns={
        "Location Type": "Location", "Description": "RM name", "Component": "RM code",
        "Batch Nr.1": "Batch number"
    }, inplace=True)

    # Truncate RM name to ensure it fits on one line in the PDF
    df_final['RM name'] = df_final['RM name'].str[:20]

    # Step 2: Create multi-block layout with category headers
    formatted_rows = []
    for location in LOCATION_ORDER:
        subset = df_final[df_final["Location"] == location]
        if not subset.empty:
            header_row = {
                "Location": location, "Location Description": "", "RM name": "", "RM code": "",
                "Batch number": "", "Available Quantity": "", "Quantity required": "",
                "Expiry Status": "", "Needs Highlighting": False
            }
            formatted_rows.append(header_row)
            subset = subset.sort_values(
                by=['Location Description', 'RM code', 'Batch number'])
            for _, row in subset.iterrows():
                formatted_rows.append(row.to_dict())

    if not formatted_rows:
        return pd.DataFrame()

    df_output = pd.DataFrame(formatted_rows)
    # Step 3: Clean up data for display
    df_output.loc[df_output['Location'] == 'Tower', 'Location Description'] = df_output.loc[df_output['Location']
                                                                                            == 'Tower', 'Location Description'].str.replace('TOWER ', '')
    return df_output


def process_data(df):
    """
    Processes the raw DataFrame to extract metadata, handle expiry dates,
    and categorize components by priority level based on cumulative quantity needed.
    """
    # Step 1: Extract metadata and Production Date
    first_row = df.iloc[0]
    production_date = pd.to_datetime(
        first_row["Current date marked the beginning"], format='%d%m%Y', errors='coerce')
    product_info = {
        "Production Ticket Nr": first_row["Production Ticket Nr"], "Wording": first_row["Wording"],
        "Product Code": first_row["Product Code"], "Quantity Launched": df["Quantity launched Theoretical"].astype(float).max(),
        "Production Date": production_date.date(),
    }
    unique_materials = df.drop_duplicates(subset=["Component", "Description"])
    product_info["Quantity Produced"] = unique_materials["Quantity required"].astype(
        float).sum()
    product_info["Raw Material Count"] = df["Component"].nunique()

    # Step 2: Filter, clean data, and handle DLUO
    # , "ZSIN", "SINF" can add as filter
    df_sing = df[df["depot location"].isin(["SING"])].copy()
    df_sing['DLUO_dt'] = pd.to_datetime(
        df_sing['DLUO'], format='%d%m%Y', errors='coerce')

    # Calculate Expiry Status
    one_month_later = production_date + pd.DateOffset(months=1)
    conditions = [
        df_sing['DLUO_dt'] < production_date,
        (df_sing['DLUO_dt'] >= production_date) & (
            df_sing['DLUO_dt'] < one_month_later)
    ]
    choices = ['Expired', 'Expiring Soon']
    df_sing['Expiry Status'] = pd.Series(pd.NA, index=df_sing.index)
    df_sing['Expiry Status'] = df_sing['Expiry Status'].fillna(
        pd.Series(np.select(conditions, choices, default='OK'), index=df_sing.index))

    # Step 3: Classify locations

    def classify_location(desc, qty_required):
        desc = str(desc).upper()
        if "HEAT" in desc:
            return "Heat"
        elif "POWDER" in desc:
            return "Powder"
        elif "TOWER" in desc and qty_required < QTY_TOWER:
            return "Tower"
        elif "TOWER" in desc:
            return "Tower overweight"
        elif desc.startswith("M") and qty_required < QTY_M:
            return "M"
        elif desc.startswith("M"):
            return "M overweight"
        elif not desc.startswith("W"):
            return "Production" if "DEFAULT" not in desc else "Default Production"
        else:
            return "Warehouse"

    df_sing["Quantity required"] = pd.to_numeric(
        df_sing["Quantity required"], errors='coerce').fillna(0)
    df_sing["Available Quantity"] = pd.to_numeric(df_sing["Available Quantity"].astype(
        str).str.replace(',', '.'), errors='coerce').fillna(0)
    df_sing["Location Type"] = df_sing.apply(lambda row: classify_location(
        row["Location Description"], row["Quantity required"]), axis=1)

    # Step 4: Assign location priority values
    location_priority = {"Heat": 1, "Powder": 2, "Tower": 3, "M": 4, "Default Production": 5,
                         "Production": 6, "Warehouse": 7, "Tower overweight": 8, "M overweight": 8}
    df_sing["Location Priority"] = df_sing["Location Type"].map(
        location_priority)

    # Step 5: Sort all potential batches
    df_sing.sort_values(
        by=["Component", "Location Priority", "Batch Nr.1"], inplace=True)
    df_sing.loc[df_sing['Location Description'] == 'Default location', 'Location Type'] = df_sing.loc[df_sing['Location Description']
                                                                                                      == 'Default location', 'Location Type'].str.replace('Default ', '')

    # Step 6: Implement cumulative quantity and highlighting logic
    all_priority_groups = []
    for component_code, group in df_sing.groupby("Component"):
        required_qty = group["Quantity required"].iloc[0]
        collected_qty = 0
        group_copy = group.copy()
        group_copy['Assigned Priority'] = 'Leftovers'

        first_priority_indices = []
        for index, row in group_copy.iterrows():
            if collected_qty < required_qty:
                first_priority_indices.append(index)
                collected_qty += row['Available Quantity']
            else:
                break
        group_copy.loc[first_priority_indices,
                       'Assigned Priority'] = 'First Priority'

        # Highlight if more than one location is needed for the first priority pick
        group_copy['Needs Highlighting'] = len(first_priority_indices) > 1

        remaining_indices = group_copy.index.difference(first_priority_indices)
        if not remaining_indices.empty:
            group_copy.loc[remaining_indices.values[0:1],
                           'Assigned Priority'] = 'Second Priority'
        if len(remaining_indices) > 1:
            group_copy.loc[remaining_indices.values[1:2],
                           'Assigned Priority'] = 'Third Priority'

        all_priority_groups.append(group_copy)

    if not all_priority_groups:
        return product_info, {}
    df_with_priorities = pd.concat(all_priority_groups)

    # Step 7: Split and format DataFrames
    priority_dfs_raw = {p_level: df_with_priorities[df_with_priorities['Assigned Priority'] == p_level] for p_level in [
        'First Priority', 'Second Priority', 'Third Priority', 'Leftovers']}
    priority_dfs_formatted = {name: format_output_df(
        df_raw) for name, df_raw in priority_dfs_raw.items() if not df_raw.empty}
    return product_info, priority_dfs_formatted


def generate_pdf(product_info, priority_dfs, barcode_locations):
    pdf_filename = f"{product_info['Production Ticket Nr']}_picking_list.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, leftMargin=0.5*inch,
                            rightMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='LeftAlign',
               alignment=TA_LEFT, fontName=FONT_NORMAL))
    styles['Normal'].fontName = FONT_NORMAL
    styles['h1'].fontName = FONT_BOLD

    elements = []
    # --- Page 1: Production Info ---
    elements.append(Paragraph("Production Ticket Information", styles['h1']))
    elements.append(Spacer(1, 0.1*inch))
    info_data = [[key, str(value)] for key, value in product_info.items()]
    info_table = Table(info_data, colWidths=[2*inch, 5*inch])
    info_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME', (0, 0), (0, -1), FONT_BOLD),
        ('FONTNAME', (1, 0), (1, -1), FONT_NORMAL),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('GRID',
                                                 (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.2*inch))

    # --- Process All Priority Levels ---
    priority_order_pdf = ['First Priority',
                          'Second Priority', 'Third Priority', 'Leftovers']
    for idx, priority_name in enumerate(priority_order_pdf):
        if priority_name in priority_dfs:
            df_output = priority_dfs[priority_name]
            if not df_output.empty:
                if idx > 0:
                    elements.append(PageBreak())
                elements.append(Paragraph(priority_name, styles['h1']))
                elements.append(Spacer(1, 0.1*inch))

                headers = ["Location", "Location\nDescription", "RM name", "RM code\n& Barcode",
                           "Batch\nnumber", "Available\nQuantity", "Quantity\nRequired"]
                table_data = [headers]

                # Build table data with conditional barcodes
                for _, row in df_output.iterrows():
                    if row['RM code'] == '':
                        table_data.append(
                            list(row.drop(['Expiry Status', 'Needs Highlighting'])))
                        continue

                    if row['Location'] in barcode_locations:
                        barcode = code128.Code128(
                            str(row['RM code']), barHeight=0.2*inch, barWidth=0.008*inch)
                        barcode_cell = [
                            Paragraph(str(row['RM code']), styles['Normal']), Spacer(10, 0), barcode]
                    else:
                        barcode_cell = Paragraph(
                            str(row['RM code']), styles['Normal'])

                    data_row = [row['Location'], Paragraph(str(row['Location Description']), styles['Normal']), Paragraph(str(
                        row['RM name']), styles['LeftAlign']), barcode_cell, Paragraph(str(row['Batch number']), styles['Normal']), row['Available Quantity'], row['Quantity required']]
                    table_data.append(data_row)

                col_widths = [0.8*inch, 1*inch, 2*inch,
                              1*inch, 1*inch, 0.8*inch, 0.8*inch]
                output_table = Table(
                    table_data, colWidths=col_widths, repeatRows=1)

                # --- Dynamic Styling for Expiry and Highlighting ---
                base_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue), ('TEXTCOLOR',
                                                                       (0, 0), (-1, 0), colors.whitesmoke),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME',
                                                             (0, 0), (-1, 0), FONT_BOLD),
                    ('GRID', (0, 0), (-1, -1), 1,
                     colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    # Override for RM name column
                    ('ALIGN', (2, 1), (2, -1), 'LEFT'),
                ]

                dynamic_styles = []
                for i, row in df_output.iterrows():
                    table_row_index = i + 1  # Account for header row
                    # Location category header rows
                    if row['RM code'] == '':
                        dynamic_styles.append(
                            ('BACKGROUND', (0, table_row_index), (-1, table_row_index), colors.lightgrey))
                        dynamic_styles.append(
                            ('TEXTCOLOR', (0, table_row_index), (-1, table_row_index), colors.black))
                        dynamic_styles.append(
                            ('FONTNAME', (0, table_row_index), (0, table_row_index), FONT_BOLD))
                    else:
                        # Expiry status coloring
                        if row['Expiry Status'] == 'Expired':
                            dynamic_styles.append(
                                ('BACKGROUND', (2, table_row_index), (2, table_row_index), colors.lightcoral))
                        elif row['Expiry Status'] == 'Expiring Soon':
                            dynamic_styles.append(
                                ('BACKGROUND', (2, table_row_index), (2, table_row_index), colors.moccasin))

                        # Highlighting for multi-location picks
                        if row['Needs Highlighting']:
                            dynamic_styles.append(
                                ('BACKGROUND', (5, table_row_index), (6, table_row_index), colors.yellow))

                output_table.setStyle(TableStyle(base_style + dynamic_styles))
                elements.append(output_table)

    doc.build(elements)
    return pdf_filename


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Production Ticket Processor")

# --- Sidebar for controls ---
st.sidebar.header("PDF Generation Options")
barcode_locations_selection = st.sidebar.multiselect(
    "Generate barcodes for which locations?",
    LOCATION_ORDER,
    default=["Tower"]
)

st.write("Upload an Excel file to see the first-priority picking list and generate a full multi-page PDF with barcodes.")
uploaded_file = st.file_uploader(
    "Choose an XLS or XLSX file", type=["xls", "xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df_preview = pd.read_excel(xls, sheet_name=sheet_name)

        header_row_index = df_preview.apply(lambda row: row.astype(
            str).str.contains("Batch Nr").any(), axis=1).idxmax() + 1

        # Define all columns that should be read as strings to preserve formatting
        string_converters = {
            'Production Ticket Nr': str, 'Product Code': str, 'Batch Nr': str,
            'Current date marked the beginning': str, 'Component': str,
            'Batch Nr.1': str, 'DLUO': str
        }
        df = pd.read_excel(xls, sheet_name=sheet_name,
                           header=header_row_index, converters=string_converters)

        product_info, priority_dataframes = process_data(df)

        st.header("Production Information")
        st.json(product_info, expanded=False)

        st.header("First Priority Picking List (Preview)")
        if 'First Priority' in priority_dataframes and not priority_dataframes['First Priority'].empty:
            st.dataframe(priority_dataframes['First Priority'].drop(
                columns=['Location', 'Expiry Status', 'Needs Highlighting']))

            if st.sidebar.button("Generate Full Picking PDF"):
                with st.spinner('Creating PDF with barcodes and highlighting...'):
                    pdf_filename = generate_pdf(
                        product_info, priority_dataframes, barcode_locations_selection)
                    with open(pdf_filename, "rb") as f:
                        st.sidebar.download_button(
                            "Download PDF", f, file_name=pdf_filename, mime="application/pdf")
                    os.remove(pdf_filename)
        else:
            st.warning(
                "No first-priority items were found based on the required quantities.")
    except Exception as e:
        st.error(f"An error occurred during file processing: {e}")
        st.error("Please ensure the Excel file has a header row containing 'Batch Nr' and 'DLUO', and that the data format is as expected.")
