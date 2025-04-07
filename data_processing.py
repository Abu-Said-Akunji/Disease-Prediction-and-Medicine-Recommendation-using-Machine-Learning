import pandas as pd
import numpy as np

# 🔹 Load dataset
file_path = "data/DiseaseAndSymptoms.csv"  # Update path if needed
df = pd.read_csv(file_path)

# 🔹 Clean symptom names (trim spaces, convert to lowercase)
df = df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)

# 🔹 Extract all unique symptoms
symptom_columns = df.columns[1:]  # Exclude "Disease" column
all_symptoms = sorted(set(symptom for col in symptom_columns for symptom in df[col].dropna().unique()))

# 🔹 Create a new one-hot encoded dataset
data_rows = []

for _, row in df.iterrows():
    disease = row["Disease"]
    symptoms_present = set(row[symptom_columns].dropna().values)  # Convert to set for fast lookup

    symptom_data = {symptom: int(symptom in symptoms_present) for symptom in all_symptoms}
    symptom_data["Disease"] = disease  # Add disease name

    data_rows.append(symptom_data)

# 🔹 Convert list to DataFrame
formatted_data = pd.DataFrame(data_rows)

# 🔹 Shuffle dataset to avoid patterns
formatted_data = formatted_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 🔹 Save the formatted dataset
formatted_file_path = "data/Formatted_Disease_Symptoms.csv"
formatted_data.to_csv(formatted_file_path, index=False)

print(f"✅ Dataset formatted and saved to {formatted_file_path}")
