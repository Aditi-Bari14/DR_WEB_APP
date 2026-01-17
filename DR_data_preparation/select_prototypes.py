import pandas as pd

# Load the CSV
df = pd.read_csv("train.csv")

# Map diagnosis to class names
label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

selected = []

# Select 5 images per class
for label, class_name in label_map.items():
    class_rows = df[df["diagnosis"] == label].head(5)

    for _, row in class_rows.iterrows():
        selected.append({
            "id_code": row["id_code"],
            "class": class_name
        })

# Save selected prototypes
out_df = pd.DataFrame(selected)
out_df.to_csv("prototype_list.csv", index=False)

print("✅ prototype_list.csv created successfully")
