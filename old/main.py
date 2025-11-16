import pdfplumber, json

tables_json = []

with pdfplumber.open("datasheet.pdf") as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for t in tables:
            headers = t[0]
            for row in t[1:]:
                tables_json.append(dict(zip(headers, row)))

with open("datasheet_tables.json", "w") as f:
    json.dump(tables_json, f, indent=2)
