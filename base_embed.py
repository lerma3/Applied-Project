#%%
import base64

# Replace 'path/to/your/file.xlsx' with the actual path to your XLSX file
file_path = 'credit_transaction_template.xlsx'

# Read the XLSX file as binary data
with open(file_path, 'rb') as file:
    binary_content = file.read()

# Encode the binary content to base64
base64_content = base64.b64encode(binary_content).decode('utf-8')

print(base64_content)
# %%
