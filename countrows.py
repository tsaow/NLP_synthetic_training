import pandas as pd
import re

df = pd.read_csv('output.csv')
total_rows = len(df)
# Remove duplicates and keep only unique rows
df = df.drop_duplicates()
unique_count = len(df)
print(f"Total number of rows before deduplication: {total_rows}")
print(f"Number of rows after deduplication: {unique_count}")

# Add similarity_score column with all 1's
df['similarity_score'] = 1

# Save the deduplicated data back to a new CSV file
df.to_csv('output_deduplicated.csv', index=False)

# Shift only the clone column and wrap the last row to the first
df['clone'] = df['clone'].shift(1)
df.loc[0, 'clone'] = df['clone'].iloc[-1]  # Move last row's clone to first position
df['similarity_score'] = 0
df.to_csv('output_shifted.csv', index=False)

#combine output_shifted.csv and output_deduplicated.csv
df_shifted = pd.read_csv('output_shifted.csv')
df_deduplicated = pd.read_csv('output_deduplicated.csv')
df_combined = pd.concat([df_shifted, df_deduplicated])
df_combined.to_csv('output_combined.csv', index=False)

#remove all instances of ```python from the clone column in output_combined.csv
df_combined['clone'] = df_combined['clone'].str.replace(r'.*```python', '', regex=True, flags=re.DOTALL)
df_combined.to_csv('output_combined.csv', index=False)
