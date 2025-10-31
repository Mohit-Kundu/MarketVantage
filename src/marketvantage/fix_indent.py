with open('report_generator_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 310 (index 309) - remove extra indentation
if len(lines) > 309:
    lines[309] = '    topic = st.text_input(\n'
if len(lines) > 310:
    lines[310] = '        "Technology Topic",\n'
if len(lines) > 311:
    lines[311] = '        value="AI lip sync",\n'
if len(lines) > 312:
    lines[312] = '        help="Enter the technology or topic you want to analyze",\n'
if len(lines) > 313:
    lines[313] = '    )\n'

with open('report_generator_app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed indentation")
